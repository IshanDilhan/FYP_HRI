"""MCN Unit Tests — Model, embeddings, window, dissonance, policy, inference."""
import sys, os, torch, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcn.config import MCNConfig, INTENT_CATEGORIES
from mcn.model import MultimodalCrossModalNetwork
from mcn.embeddings import ModalityEmbedding, MultiModalEmbedder
from mcn.temporal_window import TemporalSlidingWindow
from mcn.dissonance import DissonanceLabeler
from mcn.policy_mapper import PolicyMapper
from mcn.inference import MCNInferencePipeline

def test_config():
    c = MCNConfig()
    assert c.d_model == c.d_emb + c.d_conf_proj
    assert c.seq_len == c.window_size * c.n_modalities
    print("✓ Config test passed")

def test_modality_embedding():
    c = MCNConfig()
    emb = ModalityEmbedding(c.context_vocab_size, c.d_emb, c.d_conf_proj)
    out = emb(torch.randint(0, c.context_vocab_size, (2, 12)), torch.rand(2, 12))
    assert out.shape == (2, 12, c.d_model)
    print("✓ Modality embedding test passed")

def test_model_forward():
    c = MCNConfig(); B, W = 4, c.window_size
    model = MultimodalCrossModalNetwork(c)
    inputs = {f"{m}_idx": torch.randint(0, getattr(c, f"{m}_vocab_size"), (B, W))
              for m in ["context","emotion","gesture","motion"]}
    inputs.update({f"{m}_conf": torch.rand(B, W) for m in ["context","emotion","gesture","motion"]})
    out = model(**inputs)
    assert out["intent_logits"].shape == (B, c.n_intents)
    assert out["intent_probs"].shape == (B, c.n_intents)
    assert out["conflict_logits"].shape == (B, c.n_conflicts)
    assert out["confidence"].shape == (B, 1)
    assert torch.allclose(out["intent_probs"].sum(dim=-1), torch.ones(B), atol=1e-5)
    print(f"✓ Model forward test passed (params: {model.count_parameters():,})")

def test_temporal_window():
    c = MCNConfig(); w = TemporalSlidingWindow(c)
    frame = {"environment_context": {"state": "Classroom", "confidence": 0.94},
             "facial_affect_emotion": {"state": "Sad", "confidence": 0.89},
             "skeletal_hand_gesture": {"state": "One Hand Up", "confidence": 0.93},
             "body_motion_vector": {"state": "Sitting", "confidence": 0.97}}
    for _ in range(12): w.push(frame)
    assert w.is_ready()
    t = w.get_tensor()
    assert t["context_idx"].shape == (1, 12)
    print("✓ Temporal window test passed")

def test_dissonance_labeler():
    # Angry + Waving + Walking = only emotion-gesture conflict (Walking is neutral)
    assert DissonanceLabeler.label("Angry", "Hand Waving", "Walking") == "EMOTION_GESTURE_CONFLICT"
    # Happy + Waving + Walking = no conflict (all aligned)
    assert DissonanceLabeler.label("Happy", "Hand Waving", "Walking") == "NO_CONFLICT"
    # Angry + Waving + Leaving = emotion-gesture + gesture-motion = MULTI
    assert DissonanceLabeler.label("Angry", "Hand Waving", "Leaving") == "MULTI_CONFLICT"
    # Angry + Waving + Stationary = emotion-gesture + emotion-motion = MULTI
    assert DissonanceLabeler.label("Angry", "Hand Waving", "Stationary") == "MULTI_CONFLICT"
    print("✓ Dissonance labeler test passed")

def test_policy_mapper():
    m = PolicyMapper()
    r = m.map(1024, "HELP_REQUEST", 0.94, "Classroom")
    assert r["scenario_id"] == "SCENARIO_ID_74"
    assert "proxemic_action" in r["behavioral_policy"]
    r2 = m.map(2048, "EMERGENCY", 0.95, "Narrow Hallway")
    assert r2["behavioral_policy"]["proxemic_action"] == "YIELD_TO_WALL"
    print("✓ Policy mapper test passed")

def test_inference_pipeline():
    c = MCNConfig(); model = MultimodalCrossModalNetwork(c)
    pipe = MCNInferencePipeline(model, c)
    frame = {"environment_context": {"state": "Classroom", "confidence": 0.94},
             "facial_affect_emotion": {"state": "Sad", "confidence": 0.89},
             "skeletal_hand_gesture": {"state": "One Hand Up", "confidence": 0.93},
             "body_motion_vector": {"state": "Sitting", "confidence": 0.97}}
    result = None
    for _ in range(12): result = pipe.process_frame(frame)
    assert result is not None
    assert all(k in result for k in ["frame_id","predicted_intent","scenario_id","behavioral_policy"])
    print(f"✓ Inference pipeline test passed ({result['inference_time_ms']:.1f}ms)")
    print(f"  Output: {json.dumps(result, indent=2)}")

def test_input_packet():
    c = MCNConfig(); model = MultimodalCrossModalNetwork(c)
    pipe = MCNInferencePipeline(model, c)
    pkt = {"input_packet": {"environment_context": {"state": "Classroom", "confidence": 0.94},
            "facial_affect_emotion": {"state": "Sad", "confidence": 0.89},
            "skeletal_hand_gesture": {"state": "One Hand Up", "confidence": 0.93},
            "body_motion_vector": {"state": "Sitting", "confidence": 0.97}}}
    for _ in range(12): result = pipe.process_input_packet(pkt)
    assert result is not None
    print("✓ Input packet schema test passed")

if __name__ == "__main__":
    print("=" * 60); print("MCN UNIT TESTS"); print("=" * 60)
    tests = [test_config, test_modality_embedding, test_model_forward,
             test_temporal_window, test_dissonance_labeler, test_policy_mapper,
             test_inference_pipeline, test_input_packet]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1
        except Exception as e: print(f"✗ {t.__name__} FAILED: {e}"); failed += 1
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(0 if failed == 0 else 1)
