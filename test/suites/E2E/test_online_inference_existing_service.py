"""
Online inference tests against an already running OpenAI-compatible service.

Unlike test_online_inference.py, this module does not start or stop vLLM.
It reads the current service endpoint from test/config.yaml and validates
that the live service can answer the standard prompt correctly.

To emulate the original two-phase online inference test with a single
externally managed service, we clear HBM between phases via the live
service endpoint instead of restarting the server.
"""

import os

import pytest
import requests
from common.capture_utils import export_vars
from common.config_utils import config_utils as config_instance
from common.llm_connection.LLMBase import LLMRequest
from common.llm_connection.openai_connector import OpenAIConn
from common.llm_connection.token_counter import HuggingFaceTokenizer
from common.path_utils import get_path_relative_to_test_root


def _normalize_text(text: str) -> str:
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("！", "!")
    text = text.replace("？", "?")
    text = text.replace("：", ":")
    text = text.replace("；", ";")
    return text.strip()


def _match_any_answer(output: str, answers: list[str]) -> bool:
    normalized_output = _normalize_text(output)
    return any(normalized_output == _normalize_text(answer) for answer in answers)


def _build_live_service_client() -> OpenAIConn:
    server_url = config_instance.get_nested_config("llm_connection.server_url", "")
    model = config_instance.get_nested_config("llm_connection.model", "")
    tokenizer_path = config_instance.get_nested_config("llm_connection.tokenizer_path", "")

    if not server_url:
        raise RuntimeError("llm_connection.server_url is not configured")
    if not model:
        raise RuntimeError("llm_connection.model is not configured")
    if not tokenizer_path:
        raise RuntimeError("llm_connection.tokenizer_path is not configured")

    return OpenAIConn(
        base_url=server_url,
        tokenizer=HuggingFaceTokenizer(tokenizer_path),
        model=model,
    )


def _clear_hbm_cache(phase_name: str) -> None:
    server_url = config_instance.get_nested_config("llm_connection.server_url", "")
    llm_type = config_instance.get_nested_config("llm_connection.llm_type", "")
    enable_clear_hbm = config_instance.get_nested_config(
        "llm_connection.enable_clear_hbm", True
    )

    if not enable_clear_hbm:
        print(f"[INFO] Skip clearing HBM before {phase_name}: enable_clear_hbm=false")
        return

    if llm_type == "vllm":
        reset_url = f"{server_url}/reset_prefix_cache"
    else:
        raise RuntimeError(f"Unsupported llm_type for cache clear: {llm_type}")

    print(f"[INFO] Clearing HBM before {phase_name}: {reset_url}")
    response = requests.post(reset_url, timeout=10)
    if response.status_code == 404 and llm_type == "vllm":
        pause_url = f"{server_url}/pause"
        resume_url = f"{server_url}/resume"
        print(
            "[INFO] /reset_prefix_cache is unavailable, fallback to "
            f"{pause_url}?wait_for_inflight_requests=true&clear_cache=true"
        )
        pause_response = requests.post(
            pause_url,
            params={"wait_for_inflight_requests": "true", "clear_cache": "true"},
            timeout=10,
        )
        pause_response.raise_for_status()
        resume_response = requests.post(resume_url, timeout=10)
        resume_response.raise_for_status()
        return

    response.raise_for_status()


def _load_test_prompt_and_answers() -> tuple[str, list[str]]:
    from common.common_inference_utils import load_prompt_from_file

    test_prompt, standard_answers = load_prompt_from_file(
        get_path_relative_to_test_root("suites/E2E/prompts/test_offline_inference.json")
    )
    if not standard_answers:
        pytest.fail("No standard answers found in prompt.json")
    return test_prompt, standard_answers


def _build_messages(test_prompt: str, prompt_split_ratio: float) -> tuple[list[dict], list[dict]]:
    tokenizer_path = config_instance.get_nested_config("llm_connection.tokenizer_path", "")

    from common.common_inference_utils import split_prompt_by_tokens
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_chat_template=True)
    system_content = (
        "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，"
        "用简短的语句给出答案。\n\n例如："
        "“全国美国文学研究会的第十八届年会在哪所大学举办的？”\n"
        "回答应该为：“xx大学”。\n\n"
    )

    try:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": test_prompt},
        ]
        formatted_full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    except Exception:
        formatted_full_prompt = test_prompt

    prompt_first_part, _ = split_prompt_by_tokens(
        formatted_full_prompt, tokenizer, split_ratio=prompt_split_ratio
    )

    full_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": test_prompt},
    ]
    partial_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt_first_part},
    ]
    return full_messages, partial_messages


@pytest.mark.stage(1)
@pytest.mark.feature("online_inference_existing_service")
@pytest.mark.parametrize("max_tokens", [200])
@pytest.mark.parametrize("prompt_split_ratio", [0.5])
@export_vars
def test_online_accuracy_existing_service(max_tokens: int, prompt_split_ratio: float):
    """
    Validate an already running service using the same prompt pattern as the
    online inference E2E test, without managing the server lifecycle.
    """
    test_prompt, standard_answers = _load_test_prompt_and_answers()
    full_messages, partial_messages = _build_messages(test_prompt, prompt_split_ratio)

    print("\n===== Online Accuracy Test Against Existing Service =====")
    print(f"Server: {config_instance.get_nested_config('llm_connection.server_url', '')}")
    print(f"Model: {config_instance.get_nested_config('llm_connection.model', '')}")
    print(f"Full prompt length: {len(test_prompt)} chars")
    print(f"Max tokens: {max_tokens}")
    print(f"Prompt split ratio: {prompt_split_ratio}")

    client = _build_live_service_client()
    try:
        assert client.health_check(), "Live service health check failed"
        print(f"server models: {client.list_models()}")

        # _clear_hbm_cache("baseline request 1")
        baseline_output_1 = client.chat(
            LLMRequest(messages=full_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f'Baseline output 1: "{baseline_output_1}"')

        _clear_hbm_cache("baseline request 2 (force SSD reload)")
        baseline_output_2 = client.chat(
            LLMRequest(messages=full_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f'Baseline output 2: "{baseline_output_2}"')

        _clear_hbm_cache("mixed phase partial warmup")
        partial_output = client.chat(
            LLMRequest(messages=partial_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f'Partial prompt output: "{partial_output}"')

        full_output = client.chat(
            LLMRequest(messages=full_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f'Full prompt output: "{full_output}"')
    finally:
        client.close()

    baseline_correct = _match_any_answer(
        baseline_output_1, standard_answers
    ) and _match_any_answer(baseline_output_2, standard_answers)
    if not baseline_correct:
        pytest.fail("Existing service baseline accuracy test failed")

    mixed_correct = _match_any_answer(full_output, standard_answers)
    if not mixed_correct:
        pytest.fail("Existing service mixed prompt accuracy test failed")

    return {
        "_name": "online_inference_existing_service",
        "_data": {
            "server_url": config_instance.get_nested_config("llm_connection.server_url", ""),
            "model": config_instance.get_nested_config("llm_connection.model", ""),
            "prompt_split_ratio": prompt_split_ratio,
            "max_tokens": max_tokens,
            "baseline_output_1": baseline_output_1,
            "baseline_output_2": baseline_output_2,
            "partial_output": partial_output,
            "full_output": full_output,
            "status": "passed",
        },
    }
