from scripts.check_lora import lora_check

HF = {"base": [-20.0, -18.0, -25.0], "real": [-19.5, -17.0, -24.0], "random": [-35.0, -40.0, -30.0]}


def test_agreeing_backends_pass():
    vllm = {k: [x + 0.05 for x in v] for k, v in HF.items()}
    r = lora_check({"hf": HF, "vllm": vllm}, tolerance=1.0, max_rel_diff=0.25, min_effect=2.0)
    assert r["passed"] and r["random_effect"] == (15 + 22 + 5) / 3


def test_vllm_ignoring_the_adapter_fails():
    vllm = {"base": HF["base"], "real": HF["base"], "random": HF["base"]}  # what C1 looked like
    assert not lora_check({"hf": HF, "vllm": vllm}, tolerance=1.0, max_rel_diff=0.25, min_effect=2.0)["passed"]


def test_an_adapter_without_effect_proves_nothing():
    hf = {**HF, "random": [x + 0.1 for x in HF["base"]]}
    assert not lora_check({"hf": hf, "vllm": hf}, tolerance=1.0, max_rel_diff=0.25, min_effect=2.0)["passed"]


def test_check_lora_configs_load():
    from scripts.check_lora import KEYS
    from sl.io import load_config, require_keys

    for exp in ["owl_transitive_q35_9_0", "backdoor_0"]:
        require_keys(load_config(f"experiments/{exp}/check_lora.yaml", []), KEYS, "check_lora")
