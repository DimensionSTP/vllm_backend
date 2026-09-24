import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

sys.modules["triton_python_backend_utils"] = types.ModuleType(
    "triton_python_backend_utils"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

EmbedRequest = importlib.import_module("utils.request").EmbedRequest


class EmbedRequestTest(unittest.IsolatedAsyncioTestCase):
    async def test_string_prompt_is_rendered_before_encoding(self):
        await self._check_rendered_input(
            input_value="hello",
            expected_prompt={"prompt": "hello"},
        )

    async def test_token_prompt_is_rendered_before_encoding(self):
        await self._check_rendered_input(
            input_value=[1, 2],
            expected_prompt={"prompt_token_ids": [1, 2]},
        )

    async def _check_rendered_input(self, input_value, expected_prompt):
        tensor = Mock()
        tensor.as_numpy.return_value = np.array(
            [json.dumps({"input": input_value}).encode("utf-8")]
        )
        renderer_output = {"prompt_token_ids": [10, 20]}
        renderer = AsyncMock(return_value=[renderer_output])
        response = object()
        encoder = Mock(return_value=_responses(response))
        request = EmbedRequest(
            request=object(),
            executor_callback=encoder,
            renderer_callback=renderer,
            output_dtype=np.object_,
            logger=Mock(),
        )

        with patch(
            "utils.request.pb_utils.get_input_tensor_by_name",
            side_effect=[tensor, None, None],
            create=True,
        ):
            results = [result async for result in request.execute()]

        self.assertEqual(results, [response])
        renderer.assert_awaited_once_with([expected_prompt])
        self.assertIs(encoder.call_args.args[0], renderer_output)
        self.assertEqual(encoder.call_args.args[1].task, "embed")
        self.assertEqual(encoder.call_args.args[2], request.id)


async def _responses(response):
    yield response


if __name__ == "__main__":
    unittest.main()
