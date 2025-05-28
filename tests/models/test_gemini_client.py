import unittest
from unittest.mock import patch, AsyncMock, MagicMock, call
import os

# Assuming autogen_core and google.generativeai types are available for import
from autogen_core.models import (
    UserMessage, AssistantMessage, SystemMessage, ChatCompletionResponse, 
    ComponentModel, Usage, ResponseContent, Message, ModelInfo,
)
from autogen_core.code_executor import CancellationToken

# Mock google.generativeai before importing the client
# This is a common pattern if genai is configured at module level of gemini_client
mock_genai_module = MagicMock()
# Mock the types submodule as well, as it's accessed like genai.types.GenerationConfig
mock_genai_module.types = MagicMock() 

# Mock google.api_core.exceptions if needed for error testing
mock_google_exceptions_module = MagicMock()

# Patch sys.modules for both google.generativeai and google.api_core.exceptions
# This ensures that when gemini_client.py does 'import google.generativeai as genai'
# or 'from google.api_core import exceptions as google_exceptions', it gets our mocks.
with patch.dict('sys.modules', {
    'google.generativeai': mock_genai_module,
    'google.generativeai.types': mock_genai_module.types, # Ensure types is also mocked if client uses it directly
    'google.api_core.exceptions': mock_google_exceptions_module
}):
    from magentic_ui.models.gemini_client import GeminiChatCompletionClient


class TestGeminiChatCompletionClient(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.api_key = "test_gemini_api_key"
        self.model_name = "gemini-pro"
        self.config = {
            "model": self.model_name,
            "api_key": self.api_key,
        }
        
        # Reset mocks for each test to ensure isolation
        mock_genai_module.reset_mock()
        mock_genai_module.types.reset_mock() # Reset types mock
        mock_google_exceptions_module.reset_mock()

        # Ensure genai.configure is callable on the mock
        mock_genai_module.configure = MagicMock()
        
        # Ensure GenerativeModel returns a mock that has generate_content_async
        self.mock_gemini_model_instance = AsyncMock()
        mock_genai_module.GenerativeModel.return_value = self.mock_gemini_model_instance
        
        # Mock GenerationConfig if it's used by the client
        # The client creates GenerationConfig(temperature=..., etc.)
        # So, genai.types.GenerationConfig needs to be a callable that returns a mock
        mock_genai_module.types.GenerationConfig = MagicMock(return_value=MagicMock())


    async def test_initialization_success(self):
        client = GeminiChatCompletionClient(config=self.config)
        self.assertEqual(client.model_name, self.model_name)
        mock_genai_module.configure.assert_called_once_with(api_key=self.api_key)
        mock_genai_module.GenerativeModel.assert_called_with(self.model_name)

    async def test_initialization_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True): # Ensure no env var is picked up
            with self.assertRaisesRegex(ValueError, "Gemini API key is required"):
                GeminiChatCompletionClient(config={"model": self.model_name})

    async def test_create_success_simple(self):
        client = GeminiChatCompletionClient(config=self.config)
        
        mock_api_response = MagicMock()
        # Gemini's response structure can vary. Let's assume text is directly in parts for simplicity
        # or via candidates[0].content.parts
        mock_part = MagicMock()
        mock_part.text = "Hello from Gemini"
        mock_api_response.parts = [mock_part] 
        # If the client checks candidates first, mock that structure:
        mock_candidate_content_part = MagicMock()
        mock_candidate_content_part.text = "Hello from Gemini"
        mock_candidate = MagicMock()
        mock_candidate.content = MagicMock(parts=[mock_candidate_content_part])
        mock_api_response.candidates = [mock_candidate]

        mock_api_response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=5, 
            total_token_count=15
        )
        self.mock_gemini_model_instance.generate_content_async.return_value = mock_api_response
        
        messages = [UserMessage(content="Hello")]
        
        response = await client.create(messages=messages)
        
        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(response.content.text, "Hello from Gemini")
        self.assertEqual(response.model, self.model_name)
        self.assertEqual(response.usage.prompt_tokens, 10)
        self.assertEqual(response.usage.completion_tokens, 5)
        self.assertEqual(response.usage.total_tokens, 15)
        
        self.mock_gemini_model_instance.generate_content_async.assert_called_once()
        called_args = self.mock_gemini_model_instance.generate_content_async.call_args
        self.assertEqual(called_args[1]['contents'], [{"role": "user", "parts": [{"text": "Hello"}]}])

    async def test_create_with_system_prompt(self):
        client = GeminiChatCompletionClient(config=self.config)
        mock_api_response = MagicMock()
        mock_part = MagicMock(text="System processed user hello")
        mock_api_response.parts = [mock_part]
        mock_api_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        mock_api_response.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=8, total_token_count=28)
        self.mock_gemini_model_instance.generate_content_async.return_value = mock_api_response

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello")
        ]
        response = await client.create(messages=messages)
        self.assertEqual(response.content.text, "System processed user hello")
        
        called_args = self.mock_gemini_model_instance.generate_content_async.call_args
        expected_gemini_messages = [
            {"role": "user", "parts": [{"text": "You are a helpful assistant.\n\nHello"}]}
        ]
        self.assertEqual(called_args[1]['contents'], expected_gemini_messages)

    async def test_dump_and_load_component(self):
        original_client = GeminiChatCompletionClient(config=self.config)
        component_model = original_client.dump_component()

        self.assertEqual(component_model.provider, "magentic_ui.models.gemini_client.GeminiChatCompletionClient")
        self.assertEqual(component_model.config["model"], self.model_name)
        self.assertEqual(component_model.config["api_key"], self.api_key)

        # Reset configure mock before loading, as load_component will call __init__
        mock_genai_module.configure.reset_mock()
        mock_genai_module.GenerativeModel.reset_mock()

        loaded_client = GeminiChatCompletionClient.load_component(component_model)
        
        self.assertIsInstance(loaded_client, GeminiChatCompletionClient)
        self.assertEqual(loaded_client.model_name, self.model_name)
        self.assertEqual(loaded_client.api_key, self.api_key)
        mock_genai_module.configure.assert_called_once_with(api_key=self.api_key)
        mock_genai_module.GenerativeModel.assert_called_with(self.model_name)


    async def test_cost_calculation(self):
        client = GeminiChatCompletionClient(config=self.config) # uses gemini-pro by default in setUp
        cost = client.calculate_cost(prompt_tokens=1000, completion_tokens=1000)
        # Default gemini-pro pricing: input $0.000125/1k tokens, output $0.000375/1k tokens
        self.assertAlmostEqual(cost, (1000 * 0.000000125) + (1000 * 0.000000375))

        flash_config = {"model": "gemini-1.5-flash", "api_key": "test_key"}
        flash_client = GeminiChatCompletionClient(config=flash_config)
        flash_cost = flash_client.calculate_cost(prompt_tokens=1000, completion_tokens=1000)
        # gemini-1.5-flash pricing: input $0.000035/1K tokens, output $0.000105/1K tokens
        expected_flash_cost = (1000 * 0.000000035) + (1000 * 0.000000105)
        self.assertAlmostEqual(flash_cost, expected_flash_cost)
        
        # Test for a model not explicitly listed, should use default gemini-pro rates
        other_pro_config = {"model": "gemini-pro-somethingelse", "api_key": "test_key"}
        other_pro_client = GeminiChatCompletionClient(config=other_pro_config)
        other_pro_cost = other_pro_client.calculate_cost(prompt_tokens=1000, completion_tokens=1000)
        self.assertAlmostEqual(other_pro_cost, (1000 * 0.000000125) + (1000 * 0.000000375))

    async def test_create_api_error(self):
        # Setup the mock for GoogleAPIError. 
        # Assuming Gemini client might raise google_exceptions.GoogleAPIError or a subclass
        mock_google_exceptions_module.GoogleAPIError = Exception # Base Exception for simplicity
        mock_google_exceptions_module.InternalServerError = type('InternalServerError', (Exception,), {})

        self.mock_gemini_model_instance.generate_content_async.side_effect = mock_google_exceptions_module.InternalServerError("Test API Error")
        
        client = GeminiChatCompletionClient(config=self.config)
        messages = [UserMessage(content="Hello")]
        
        with self.assertRaises(mock_google_exceptions_module.InternalServerError):
            await client.create(messages=messages)

    async def test_message_conversion_multiple_roles(self):
        client = GeminiChatCompletionClient(config=self.config) # Not used for API call
        
        messages = [
            SystemMessage(content="Be concise."),
            UserMessage(content="Hi"),
            AssistantMessage(content="Hello!"),
            UserMessage(content="How are you?"),
            # Example of a message that might not be just a string
            UserMessage(content={"type": "text", "text": "Complex content"}) 
        ]
        
        # This test focuses on the message transformation logic, which is internal to create()
        # We can assert the 'contents' argument passed to generate_content_async
        mock_api_response = MagicMock() # Dummy response
        mock_api_response.parts = [MagicMock(text="response")]
        mock_api_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="response")]))]
        mock_api_response.usage_metadata = MagicMock(prompt_token_count=1, candidates_token_count=1, total_token_count=2)
        self.mock_gemini_model_instance.generate_content_async.return_value = mock_api_response

        await client.create(messages=messages)

        called_args = self.mock_gemini_model_instance.generate_content_async.call_args
        actual_contents = called_args[1]['contents']
        
        expected_contents = [
            # System prompt prepended to first user message
            {"role": "user", "parts": [{"text": "Be concise.\n\nHi"}]}, 
            {"role": "model", "parts": [{"text": "Hello!"}]},
            {"role": "user", "parts": [{"text": "How are you?"}]},
            {"role": "user", "parts": [{"text": "{'type': 'text', 'text': 'Complex content'}"}]} # Stringified
        ]
        self.assertEqual(actual_contents, expected_contents)

    async def test_model_info(self):
        client = GeminiChatCompletionClient(config=self.config)
        model_info = client.model_info
        self.assertIsInstance(model_info, ModelInfo)
        self.assertEqual(model_info.family, "Gemini")
        self.assertEqual(model_info.name, self.model_name)
        self.assertEqual(model_info.context_length, 30720) # Default from client
        self.assertEqual(model_info.max_output_tokens, 2048) # Default from client
        self.assertFalse(model_info.vision)

        vision_config = {"model": "gemini-pro-vision", "api_key": "test_key"}
        vision_client = GeminiChatCompletionClient(config=vision_config)
        vision_model_info = vision_client.model_info
        self.assertTrue(vision_model_info.vision)

    async def test_create_with_kwargs(self):
        client = GeminiChatCompletionClient(config=self.config)
        mock_api_response = MagicMock()
        mock_api_response.parts = [MagicMock(text="Response with kwargs")]
        mock_api_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="Response with kwargs")]))]
        mock_api_response.usage_metadata = MagicMock(prompt_token_count=1, candidates_token_count=1, total_token_count=2)
        self.mock_gemini_model_instance.generate_content_async.return_value = mock_api_response

        messages = [UserMessage(content="Test kwargs")]
        await client.create(
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            top_p=0.8,
            top_k=40,
            stop=["\n", "stop"]
        )

        self.mock_gemini_model_instance.generate_content_async.assert_called_once()
        called_args = self.mock_gemini_model_instance.generate_content_async.call_args
        
        # The client creates a GenerationConfig object. We need to check its attributes.
        # genai.types.GenerationConfig was mocked to return a MagicMock instance.
        # We need to assert that this mock was called with the correct parameters.
        mock_genai_module.types.GenerationConfig.assert_called_once_with(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=100,
            stop_sequences=["\n", "stop"]
        )
        # And this returned mock instance should be what's passed to generate_content_async
        self.assertEqual(called_args[1]['generation_config'], mock_genai_module.types.GenerationConfig.return_value)

    async def test_create_with_cancellation_token(self):
        client = GeminiChatCompletionClient(config=self.config)
        cancellation_token = CancellationToken()
        cancellation_token.cancel() # Pre-cancel

        response = await client.create(messages=[UserMessage(content="Hello")], cancellation_token=cancellation_token)

        self.assertIsNone(response.content) # Or ResponseContent(text="") depending on client's exact behavior
        self.assertEqual(response.usage.total_tokens, 0)
        self.mock_gemini_model_instance.generate_content_async.assert_not_called()
        
    async def test_empty_messages_list_create(self):
        client = GeminiChatCompletionClient(config=self.config)
        response = await client.create(messages=[])
        self.assertEqual(response.content.text, "")
        self.assertEqual(response.usage.total_tokens, 0)
        self.mock_gemini_model_instance.generate_content_async.assert_not_called()


if __name__ == '__main__':
    unittest.main()
