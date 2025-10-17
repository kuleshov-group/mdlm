import torch
from transformers import GPT2Tokenizer
from diffusion import Diffusion

def test_q_xt():
    # Setup: create a dummy config and tokenizer
    class DummyConfig:
        class Sampling:
            predictor = 'ddpm'
        class Eval:
            gen_ppl_eval_model_name_or_path = 'gpt2'
        class Training:
            antithetic_sampling = False
            importance_sampling = False
            change_of_variables = False
            ema = 0
            sampling_eps = 0.001
        class Model:
            length = 5
            hidden_size = 8         # Add this line
            cond_dim = 4            # Add this line
            n_blocks = 1            # Add this line
            n_heads = 1             # Add this line
            scale_by_sigma = False  # Add this line
            dropout = 0.0           # Add this line
            tie_word_embeddings = False # Add this line
        class Loader:
            eval_batch_size = 2
        class Noise:
            type = 'loglinear'
            sigma_min = 0.0001
            sigma_max = 20
        sampling = Sampling()
        eval = Eval()
        training = Training()
        model = Model()
        loader = Loader()
        noise = Noise()
        backbone = 'dit'
        parameterization = 'subs'
        T = 0
        subs_masking = False
        time_conditioning = False
        optim = type('optim', (), {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'weight_decay': 0})()
        lr_scheduler = type('lr_scheduler', (), {})()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    config = DummyConfig()
    model = Diffusion(config, tokenizer)

    # Test input
    x = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    move_chance = torch.tensor(0.5)  # 50% chance to flip

    # Run q_xt
    xt = model.q_xt(x, move_chance)

    print("Original x:\n", x)
    print("Noised xt:\n", xt)
    print("Changed positions:\n", (xt != x))

    # Check: all values in xt are valid vocab indices
    assert ((xt >= 0) & (xt < model.vocab_size)).all(), "Invalid token index in xt"

    # Edge case: move_chance = 0 (no change)
    xt0 = model.q_xt(x, torch.tensor(0.0))
    assert torch.equal(xt0, x), "move_chance=0 should not change x"

    # Edge case: move_chance = 1 (all change)
    xt1 = model.q_xt(x, torch.tensor(1.0))
    assert (xt1 != x).all(), "move_chance=1 should change all positions"

    print("All tests passed.")

if __name__ == "__main__":
    test_q_xt()