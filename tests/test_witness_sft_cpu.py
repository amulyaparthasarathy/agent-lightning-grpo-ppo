"""Real, locally initialized Llama. Requires the locked tokenizer file and CPU environment."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiment import recovery_prompt, reader_assets
from witness_sft import artifact, data, training


class CPUTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tok = training.tokenizer(Path(os.environ['WITNESS_TOKENIZER']))

    def model(self):
        torch.manual_seed(12345)  # initialization of the CPU fixture only
        return LlamaForCausalLM(LlamaConfig(vocab_size=128256, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True,
            bos_token_id=training.BOS, eos_token_id=training.EOT, attn_implementation='sdpa'))

    def test_pinned_masks_and_rejection(self):
        tokens = []
        for demo in data.load():
            ids, labels = training.encode(demo, self.tok)
            prompt = recovery_prompt.render(demo['messages'], recovery_prompt.tool_definitions())
            context = self.tok.encode(prompt, add_special_tokens=False).ids
            self.assertEqual(labels[:len(context)], [training.IGNORE] * len(context))
            target = self.tok.encode(demo['target'], add_special_tokens=False).ids + [training.EOT]
            self.assertEqual(ids, context + target)
            self.assertEqual(labels[len(context):], target)
            tokens.append(len(target))
        self.assertEqual(sum(tokens), 2540)
        demo['messages'][0]['content'] += ' word' * 2048
        with self.assertRaises(ValueError):
            training.encode(demo, self.tok)
        with self.assertRaises(ValueError):
            training.tokenizer(Path(__file__))

    def test_loss_and_gradients_match_dense_reference(self):
        model = self.model()
        batch = training.collate([([1, 2, 3, 4, 5], [-100, -100, 3, 4, 5]),
                                  ([2, 3, 4], [-100, -100, 4])], 'cpu')
        ids, labels, attention = batch
        self.assertEqual(labels[1].tolist(), [-100, -100, 4, -100, -100])
        self.assertEqual(attention[1].tolist(), [1, 1, 1, 0, 0])
        actual = training.loss(model, batch)
        logits = model(ids, attention_mask=attention, use_cache=False).logits.log_softmax(-1)
        expected = -(logits[0, [1, 2, 3], [3, 4, 5]].mean() + logits[1, 1, 4]) / 2
        torch.testing.assert_close(actual, expected)
        params = list(training.projections(model).values())
        for a, b in zip(torch.autograd.grad(actual, params), torch.autograd.grad(expected, params)):
            torch.testing.assert_close(a, b)

    def test_one_pass_save_restore(self):
        model = self.model()
        base = copy.deepcopy(model)
        binding = artifact.identity(artifact.CPU, '0'*40, {'fixture': 'cpu'}, artifact.fixture_base(base))
        stats = training.fit(model, self.tok, torch.device('cpu'))
        self.assertEqual(stats['updates'], 8)
        self.assertEqual(stats['supervised_tokens'], 2540)
        self.assertTrue(all(0 < n < float('inf') for n in stats['preclip_norms']))
        changed = {k for k, v in model.state_dict().items() if not torch.equal(v, base.state_dict()[k])}
        self.assertEqual(changed, set(training.projections(model)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'artifact'
            digest = artifact.save(path, model, binding, stats)
            def restore(expected=binding, commit=digest):
                return artifact.restore(path, base, expected, commit, {'fixture': 'cpu-reload'})
            for bad in ({'class': artifact.CPU}, {**binding, 'source': '1'*40}, {**binding, 'recipe': {}},
                        {**binding, 'manifest_sha256': '0'*64}):
                with self.assertRaises(ValueError):
                    restore(bad)
            with self.assertRaises(ValueError):
                restore(commit='0'*64)
            live = artifact.identity(artifact.LIVE, binding['source'], artifact.worker('fc-train', 'ta-train'),
                                     reader_assets.model_lock()['config.json']['sha256'])
            for consumer in [live['producer'], artifact.worker('fc-new', 'ta-train'), artifact.worker('fc-new', 'ta-new')]:
                with self.assertRaises(ValueError):
                    artifact.restore(path, base, live, digest, consumer)
            restore()
            model.eval()
            ids = torch.tensor([[training.BOS, 2, 3]])
            with torch.no_grad():
                self.assertTrue(torch.equal(model(ids).logits, base(ids).logits))
            for name, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, base.state_dict()[name]))
            with torch.no_grad():
                next(base.parameters()).add_(0.01)
            with self.assertRaises(ValueError):
                restore()
            (path / artifact.WEIGHTS).unlink()
            with self.assertRaises(ValueError):
                artifact.restore(path, model, binding, digest, {'fixture': 'cpu-reload'})
        print(f"CPU fixture: {stats['examples']} examples; {stats['updates']} updates; "
              f"preclip norms {min(stats['preclip_norms']):.8g}..{max(stats['preclip_norms']):.8g}; exact reload")


if __name__ == '__main__':
    unittest.main()
