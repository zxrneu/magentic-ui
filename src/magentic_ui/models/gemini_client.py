import os
from typing import Any, List, Dict, Optional, Union

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from autogen_core.models import (
    ChatCompletionClient,
    Message, 
    ChatCompletionResponse, 
    ModelInfo,
    Usage,
    ResponseContent,
    # FunctionCall, # Not used in initial version
    # ToolCall, # Not used in initial version
    ComponentModel,
)
from autogen_core.code_executor import CancellationToken

# Assuming autogen_core.models.Message is the base class for message types
# If not, specific message types like UserMessage, SystemMessage might be needed
# from autogen_core.messages import UserMessage, AssistantMessage # Example if Message is not a concrete class

class GeminiChatCompletionClient(ChatCompletionClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config) # Pass the whole config
        self.config = config # Store config for dump_component
        self.model_name = config.get("model", "gemini-pro")
        self.api_key = config.get("api_key", os.environ.get("GEMINI_API_KEY"))
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set it in the config or GEMINI_API_KEY environment variable.")
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)

    async def create(
        self,
        messages: List[Message], 
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        gemini_messages = []
        system_prompt = None

        # Convert autogen_core messages to Gemini format
        # Gemini expects a list of {"role": "user/model", "parts": [{"text": "content"}]}
        # System prompts are handled differently, often as a top-level parameter to GenerativeModel
        # or as the first part of the conversation with a specific structure.
        # For this initial pass, we will extract the last system message if present.
        
        temp_messages = []
        for msg in messages:
            role = getattr(msg, 'role', 'user') # Default to user if no role
            content = getattr(msg, 'content', '')
            
            if role == "system":
                system_prompt = content # Store the last system prompt
                continue # Don't add system prompts to the main message list for generate_content

            # Convert role names if necessary. Gemini uses 'user' and 'model'.
            if role == "assistant":
                role = "model"
            
            # Ensure content is a string, as Gemini parts expect text.
            if not isinstance(content, str):
                # Handle cases where content might be a list of parts (e.g. for multimodal)
                # For now, assuming content should be stringifiable.
                content = str(content)

            temp_messages.append({"role": role, "parts": [{"text": content}]})
        
        # Gemini's `generate_content` expects alternating user/model roles.
        # Ensure the sequence is valid. The SDK might handle some corrections, but it's good practice.
        # For now, we pass the converted messages as `contents`.
        # If there's a system prompt, it's passed to GenerativeModel initialization if possible,
        # or as part of the first message in some Gemini interaction patterns.
        # The current `genai.GenerativeModel` takes `system_instruction` at init.
        # We might need to re-initialize the model if system_prompt changes, or use a chat session.
        
        # For simplicity in this version, if a system_prompt was found,
        # we will prepend it to the user's first message if the first message is from a user.
        # This is a common workaround if the SDK/model doesn't directly support system messages in `generate_content` history.
        # A more robust solution would involve using `model.start_chat()` and `chat.send_message()`
        # or initializing `GenerativeModel` with `system_instruction` if it's static.
        
        final_gemini_messages = []
        first_user_message_processed = False
        if system_prompt and temp_messages and temp_messages[0]['role'] == 'user':
            # Prepend system prompt to the first user message's content
            original_content = temp_messages[0]['parts'][0]['text']
            # This is a simplistic merge. Consider how multi-part messages would be handled.
            final_gemini_messages.append({
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\n{original_content}"}]
            })
            final_gemini_messages.extend(temp_messages[1:])
            first_user_message_processed = True
        else:
            final_gemini_messages = temp_messages
            if system_prompt:
                 # If no user message to prepend to, this system prompt might be lost
                 # or needs to be handled by initializing the model differently.
                 # For now, we'll log if a system prompt is present but not used.
                 # print(f"Warning: System prompt provided but not directly used in this request structure: {system_prompt}")
                 pass


        generation_config_params = {
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "max_output_tokens": kwargs.get("max_tokens"),
            # "stop_sequences": kwargs.get("stop"), # Needs to be a list of strings
        }
        if "stop" in kwargs and isinstance(kwargs["stop"], list):
            generation_config_params["stop_sequences"] = kwargs["stop"]

        generation_config_params = {k: v for k, v in generation_config_params.items() if v is not None}
        generation_config = GenerationConfig(**generation_config_params) if generation_config_params else None

        try:
            if cancellation_token and cancellation_token.is_cancelled:
                return ChatCompletionResponse(
                    content=None, model=self.model_name, usage=Usage(total_tokens=0), cost=0.0
                )

            # Using generate_content_async for non-streaming.
            # Ensure final_gemini_messages is not empty.
            if not final_gemini_messages:
                return ChatCompletionResponse(
                    content=ResponseContent(text=""), model=self.model_name, usage=Usage(total_tokens=0), cost=0.0
                )

            response = await self._model.generate_content_async(
                contents=final_gemini_messages, # Corrected variable name
                generation_config=generation_config,
            )

            response_content_str = ""
            if response.parts: # This is more common for non-candidate responses
                response_content_str = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 response_content_str = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            
            # Token counting: Gemini API response usually includes token counts in metadata or usage fields.
            # Example: response.usage_metadata (prompt_token_count, candidates_token_count, total_token_count)
            prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
            completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0 # Or response.usage_metadata.total_token_count - prompt_tokens
            total_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else prompt_tokens + completion_tokens
            
            usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            return ChatCompletionResponse(
                content=ResponseContent(text=response_content_str),
                model=self.model_name,
                usage=usage,
                cost=cost,
                raw_response=response, # Store raw response if needed
            )

        except Exception as e:
            # Consider more specific error handling for Gemini exceptions
            # from google.api_core import exceptions as google_exceptions
            # if isinstance(e, google_exceptions.GoogleAPIError):
            #     # Handle Google API specific errors
            # else:
            #     # Handle other errors
            # For now, just re-raising
            raise e


    def dump_component(self) -> ComponentModel:
        # Ensure the provider string is something ChatCompletionClient.load_component can resolve.
        # This might mean this class needs to be registered or be in a known path.
        return ComponentModel(
            provider=f"magentic_ui.models.gemini_client.{self.__class__.__name__}", 
            config=self.config, # Use the stored config
        )

    @classmethod
    def load_component(cls, component_model: ComponentModel) -> "GeminiChatCompletionClient":
        return cls(config=component_model.config)

    @property
    def model_info(self) -> ModelInfo:
        # These are examples, refer to official Gemini documentation for accurate, model-specific values.
        # Context length for "gemini-pro" is 30720 tokens (text only) or 12288 for vision model inputs.
        # Max output tokens for "gemini-pro" is 2048.
        return ModelInfo(
            family="Gemini",
            name=self.model_name,
            context_length=30720, 
            max_output_tokens=2048, # Example, check specific model
            token_cost_info=None, 
            vision="vision" in self.model_name.lower(), # Basic check if model name suggests vision
        )

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        # Placeholder for Gemini pricing. E.g., for gemini-pro:
        # Input: $0.000125 / 1K characters (or token equivalent)
        # Output: $0.000375 / 1K characters (or token equivalent)
        # Gemini API uses characters for some models, tokens for others. Assuming tokens for now.
        # This needs to be updated with actual pricing for the specific model_name.
        # For "gemini-1.0-pro" (pay-as-you-go):
        # Input: $0.000125 / 1k tokens -> $0.000000125 per token
        # Output: $0.000375 / 1k tokens -> $0.000000375 per token
        cost_per_prompt_token = 0.000000125 
        cost_per_completion_token = 0.000000375
        
        if "gemini-1.5" in self.model_name: # Example for a different model family
            # gemini-1.5-flash (on-demand)
            # Input: $0.000125 / 1K tokens (for <128K context), $0.00025 / 1K tokens (for >=128K context)
            # Output: $0.000375 / 1K tokens (for <128K context), $0.00075 / 1K tokens (for >=128K context)
            # This example doesn't implement context window pricing.
            cost_per_prompt_token = 0.000125 / 1000
            cost_per_completion_token = 0.000375 / 1000
            if "flash" in self.model_name: # gemini-1.5-flash specific
                 cost_per_prompt_token = 0.000035 / 1000 # Based on $0.035 per 1M tokens for flash
                 cost_per_completion_token = 0.000105 / 1000 # Based on $0.105 per 1M tokens for flash
        
        cost = (prompt_tokens * cost_per_prompt_token) + (completion_tokens * cost_per_completion_token)
        return cost
