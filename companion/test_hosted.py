"""Offline contracts. Prepare the pinned public tokenizer before running this module."""
from concurrent.futures import Future
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import create_autospec

import httpx
import numpy as np

import tinker

from companion import hosted as h
from companion.grpo import Episode, Turn, training_batch


def ready(value):
    future = Future()
    future.set_result(value)
    return future


def group():
    result = []
    for i, reward in enumerate([0, 0, 1, 1]):
        # Token 777 is a tool observation in the second prompt.
        turns = [Turn("v0", f"{i}/0", [31, 32, 33], [42, 43], [-0.3, -0.7], "stop")]
        if i % 2:
            turns.append(Turn("v0", f"{i}/1", [31, 42, 43, 777, 33], [44], [-0.4], "stop"))
        result.append(Episode(f"case/{i}", "case", reward, turns))
    return result


class BatchTests(unittest.TestCase):
    def test_exact_shift_mask_and_episode_mass(self):
        episodes = group()
        rows, trace, informative = training_batch(episodes, "case", "v0")
        self.assertTrue(informative)
        self.assertEqual(len(rows), 6)
        for row, item in zip(rows, trace, strict=True):
            i, j = map(int, item["sequence_id"].split("/"))
            turn = episodes[i].turns[j]
            self.assertEqual(row.model_input.to_ints(), (turn.prompt + turn.tokens)[:-1])
            fields = {k: v.tolist() for k, v in row.loss_fn_inputs.items()}
            self.assertEqual(fields["target_tokens"], (turn.prompt + turn.tokens)[1:])
            self.assertEqual(item["mask"], [0] * (len(turn.prompt)-1) + [1] * len(turn.tokens))
            expected = (episodes[i].reward - 0.5) / (math.sqrt(1/3) + 1e-6)
            self.assertAlmostEqual(item["advantage"], expected)
            for mask, lp, scaled in zip(item["mask"], fields["logprobs"], fields["advantages"], strict=True):
                if not mask:
                    self.assertEqual((lp, scaled), (0, 0))
                else:
                    self.assertAlmostEqual(scaled, expected * item["weight_per_generated_token"])
            for actual, old in zip(fields["logprobs"][-len(turn.tokens):], turn.logprobs, strict=True):
                self.assertAlmostEqual(actual, old)
        for episode in episodes:
            mass = sum(sum(x["mask"]) * x["weight_per_generated_token"]
                       for x in trace if x["episode_id"] == episode.episode_id)
            self.assertAlmostEqual(mass, 0.25)

    def test_constant_groups(self):
        for reward in (0, 1):
            batch, _, informative = training_batch([replace(e, reward=reward) for e in group()], "case", "v0")
            self.assertFalse(informative)
            self.assertTrue(all(set(row.loss_fn_inputs["advantages"].tolist()) == {0} for row in batch))

    def test_reject_mixed_and_incomplete_evidence(self):
        a, b, *rest = group()
        for bad in (replace(a, episode_id=b.episode_id), replace(a, case_id="other"),
                    replace(a, reward=None), replace(a, turns=[])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                training_batch([bad, b, *rest], "case", "v0")
        for change in (dict(policy="v1"), dict(sequence_id=b.turns[0].sequence_id),
                       dict(logprobs=[]), dict(logprobs=[math.nan, 0]), dict(stop="length")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                training_batch([replace(a, turns=[replace(a.turns[0], **change)]), b, *rest], "case", "v0")


class Fixture:
    """SDK signatures, response types, tokenizer and task are real. All service calls are local doubles."""

    run_id = "fixture:train:0"
    source = "fixture-source"

    def __init__(self, out, *, constant=False):
        self.tok = h.tokenizer()
        self.out = out
        self.constant = constant
        self.service = create_autospec(tinker.ServiceClient, instance=True)
        self.rest = create_autospec(tinker.RestClient, instance=True)
        self.trainer = create_autospec(tinker.TrainingClient, instance=True)
        self.trainer.get_info.return_value = tinker.types.GetInfoResponse(
            model_data={"model_name": h.MODEL}, model_id=self.run_id, is_lora=True, lora_rank=h.RANK)
        self.service.create_lora_training_client.return_value = ready(self.trainer)
        self.trainer.save_weights_for_sampler.side_effect = self.save
        self.trainer.forward_backward.side_effect = self.backward
        self.trainer.optim_step.return_value = ready(tinker.types.OptimStepResponse(metrics={}))
        self.service.create_sampling_client.side_effect = self.load
        self.rest.list_checkpoints.side_effect = self.list_checkpoints
        self.samples = 0

    def list_checkpoints(self, run_id):
        assert run_id == self.run_id
        rows = [json.loads(p.read_text()) for p in self.out.glob("fixture-v*.json")]
        return ready(tinker.types.CheckpointsListResponse(checkpoints=rows))

    def save(self, name, ttl_seconds, user_metadata):
        step = self.trainer.optim_step.call_count
        assert name == f"asset-v{step}" and ttl_seconds == h.TTL
        path = h.checkpoint_path(self.run_id, step)
        now = datetime.now(timezone.utc)
        item = tinker.Checkpoint(tinker_path=path, checkpoint_type="sampler", checkpoint_id=name,
            time=now, expires_at=now + timedelta(seconds=ttl_seconds), size_bytes=4096,
            base_model=h.MODEL, user_metadata=user_metadata)
        h.write(self.out / f"fixture-v{step}.json", item.model_dump(mode="json"))
        return ready(tinker.types.SaveWeightsForSamplerResponse(path=path))

    def backward(self, batch, loss_fn, loss_fn_config):
        assert len(batch) == 8 and loss_fn == "ppo"
        assert loss_fn_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
        outputs = [{"logprobs": row.loss_fn_inputs["logprobs"]} for row in batch]
        return ready(tinker.ForwardBackwardOutput("fixture", outputs, {"loss:sum": 0.0}))

    def load(self, model_path, retry_config):
        assert not retry_config.enable_retry_logic
        assert model_path in {c.tinker_path for c in self.list_checkpoints(self.run_id).result().checkpoints}
        client = create_autospec(tinker.SamplingClient, instance=True)
        client._get_sampler_submit.return_value = ready(tinker.types.GetSamplerResponse(
            sampler_id="fixture-sampler", model_path=model_path, base_model=h.MODEL))
        client.sample.side_effect = self.sample
        return ready(client)

    def sample(self, prompt, num_samples, sampling_params):
        self.samples += 1
        assert num_samples == 1
        rendered = self.tok.decode(prompt.to_ints())
        assert rendered.endswith("<think>\n\n</think>\n\n")
        asset = re.search(r"Look up asset (asset-\d{3})", rendered)[1]
        arguments = {"asset_id": asset}
        if "Lookup result:" in rendered:
            answer = h.task.dumps(h.task.lookup(arguments))
            if sampling_params.temperature and self.samples % 4 == 0 and not self.constant:
                answer = "{}"
        else:
            assert h.task.dumps(h.task.lookup(arguments)) not in rendered
            answer = h.task.dumps(arguments)
        ids = self.tok.encode(answer, add_special_tokens=False) + [self.tok.eos_token_id]
        seq = tinker.SampledSequence("stop", f"fixture-{self.samples}", np.array(ids), np.full(len(ids), -0.5))
        return ready(tinker.SampleResponse(sequences=[seq]))


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_locked_sdk_submissions_include_retries_and_rejections(self):
        from tinker._client import AsyncTinker

        for fail_first, expected in [(False, 4), (True, 1)]:
            with self.subTest(fail_first=fail_first):
                sent = []
                def transport(request):
                    sent.append(request.url.path)
                    return httpx.Response(503 if fail_first else 200, json={"request_id": "fixture"})

                async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
                    # All requests terminate in the in-memory transport above.
                    sdk = AsyncTinker(base_url="https://fixture.invalid", http_client=http,
                                      api_key="tml-offline-fixture", max_retries=0)
                    payload = tinker.SampleRequest(prompt=tinker.ModelInput.from_ints([42]),
                                                   sampling_params=tinker.SamplingParams(max_tokens=1))
                    journal = []
                    with h.request_limits(True, journal.append):
                        for _ in range(expected):
                            if fail_first:
                                with self.assertRaises(tinker.InternalServerError):
                                    await sdk.sampling.asample(request=payload)
                            else:
                                await sdk.sampling.asample(request=payload)
                        with self.assertRaises(tinker.APIConnectionError):
                            await sdk.sampling.asample(request=payload)
                    self.assertEqual(sent, ["/api/v1/asample"] * expected)
                    self.assertEqual(journal, ["asample"] * expected)


class LifecycleTests(unittest.TestCase):
    def test_frozen_schedule_and_independent_verifier(self):
        train, dev = h.panels()
        self.assertEqual([c["case_id"].split("/")[-1] for c in train + dev],
                         ["asset-011", "asset-041", "asset-304", "asset-231"])
        args = '{"asset_id":"asset-011"}'
        self.assertEqual(h.observe(args), h.task.lookup(json.loads(args)))
        self.assertEqual(h.observe("not JSON")["code"], "invalid_argument_json")
        self.assertEqual(h.observe('{"asset_id":"missing"}')["code"], "invalid_asset_id")
        with self.assertRaises(ValueError):
            h.task.expected(train[0]["question"], dev[0]["result"])

    def test_uniform_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            f = Fixture(out, constant=True)
            self.assertIsNone(h.collect(f.service, f.rest, f.tok, out, f.source))
            f.trainer.forward_backward.assert_not_called()
            f.trainer.optim_step.assert_not_called()
            self.assertEqual(f.samples, 12)
            self.assertFalse((out / "handoff.json").exists())
            self.assertEqual(json.loads((out / "result.json").read_text())["updates"], 0)

    def test_updates_refresh_and_new_process_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            f = Fixture(out)
            digest = h.collect(f.service, f.rest, f.tok, out, f.source)
            self.assertEqual(f.trainer.optim_step.call_count, 2)
            self.assertEqual(f.samples, 20)
            events = [json.loads(s) for s in (out / "events.jsonl").read_text().splitlines()]
            policies = [e["policy"] for e in events if e["event"] == "sample_requested"]
            self.assertEqual(policies, [h.checkpoint_path(f.run_id, 0)]*12 + [h.checkpoint_path(f.run_id, 1)]*8)
            with self.assertRaisesRegex(ValueError, "producer"):
                h.reload(f.service, f.rest, f.tok, out, f.source, digest)
            # Real interpreter boundary. Only the mock remote store and manifest survive.
            command = [sys.executable, "-m", "companion.test_hosted", tmp, digest]
            subprocess.run(command, check=True)
            result = json.loads((out / "result.json").read_text())
            self.assertNotEqual(os.getpid(), result["process"])
            self.assertEqual(result["baseline"], result["reload"])
            self.assertEqual(result["checkpoint"], h.checkpoint_path(f.run_id, 2))
            with self.assertRaisesRegex(ValueError, "manifest"):
                h.reload(f.service, f.rest, f.tok, out, f.source, "tampered")

    def test_wrong_sampler_readback_stops_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            f = Fixture(out)
            client = create_autospec(tinker.SamplingClient, instance=True)
            client._get_sampler_submit.return_value = ready(tinker.types.GetSamplerResponse(
                sampler_id="stale", base_model=h.MODEL, model_path=h.checkpoint_path(f.run_id, 1)))
            f.service.create_sampling_client.side_effect = None
            f.service.create_sampling_client.return_value = ready(client)
            with self.assertRaisesRegex(ValueError, "sampler identity"):
                h.collect(f.service, f.rest, f.tok, out, f.source)
            client.sample.assert_not_called()
            f.trainer.optim_step.assert_not_called()

    def test_checkpoint_wrong_run_kind_step_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            path = f.save("asset-v0", h.TTL, {"source": f.source, "step": "0"}).result().path
            for wrong in (path.replace(f.run_id, "different"), path.replace("sampler_weights", "weights"),
                          path.replace("asset-v0", "asset-v2")):
                with self.assertRaisesRegex(ValueError, "Wrong"):
                    h.checkpoint(f.rest, f.run_id, 0, f.source, wrong)
            with self.assertRaisesRegex(ValueError, "metadata"):
                h.checkpoint(f.rest, f.run_id, 0, "different-source", path)


if __name__ == "__main__":
    fixture = Fixture(Path(sys.argv[1]))
    h.reload(fixture.service, fixture.rest, fixture.tok, fixture.out, fixture.source, sys.argv[2])
    fixture.service.create_lora_training_client.assert_not_called()
    assert fixture.samples == 4
