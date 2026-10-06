"""Regression checks for malformed FAST bytes that previously passed the shape check."""
import importlib.util,json,unittest
from pathlib import Path
import numpy as np
from transformers import PreTrainedTokenizerFast
from fast_decode_validation import validate_fast_tokens

ROOT=Path('/home/kb/pi0_fast')
class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bpe=PreTrainedTokenizerFast(tokenizer_file=str(ROOT/'fast-tokenizer/tokenizer.json'),clean_up_tokenization_spaces=False)
        spec=importlib.util.spec_from_file_location('local_fast_test',ROOT/'fast-tokenizer/processing_action_tokenizer.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.fast=module.UniversalActionProcessor(cls.bpe,scale=10,min_token=-354,vocab_size=2048)
    def test_known_corrupt_tokens_have_correct_length_but_invalid_bytes(self):
        for ids in [[309,1555,222,339,369,439,266,759,274,287,261,864],
                    [283,344,277,248,1779,421,272,295,324,287,277,293,287,301,360,416,261,257]]:
            decoded=self.bpe.decode(ids)
            self.assertEqual(len(decoded),70);self.assertIn('\ufffd',decoded)
            self.assertGreater(np.abs(self.fast.decode([ids],time_horizon=10,action_dim=7)).max(),1000)
            with self.assertRaisesRegex(ValueError,'Malformed UTF-8'):validate_fast_tokens(self.bpe,ids,70)
    def test_valid_encodings_preserved(self):
        rng=np.random.default_rng(42)
        for actions in rng.uniform(-1,1,(30,10,7)):
            ids=self.fast(actions)[0]
            self.assertEqual(validate_fast_tokens(self.bpe,ids,70),self.bpe.decode(ids))
    def test_invalid_ids_and_shape(self):
        with self.assertRaisesRegex(ValueError,'vocabulary'):validate_fast_tokens(self.bpe,[99999],70)
        with self.assertRaisesRegex(ValueError,'coefficients'):validate_fast_tokens(self.bpe,[],70)
    def test_audited_tokenizer_fallback_and_valid_parity(self):
        from serve_audited_policy import AuditedTokenizer,last_decode
        from openpi.models.tokenizer import FASTTokenizer
        class Paligemma:
            def __init__(self,ids):self.ids=ids
            def decode(self,tokens):return 'Action: example|'
            def encode(self,text):return [262144-1-128-x for x in self.ids]
            def vocab_size(self):return 262144
        instance=object.__new__(AuditedTokenizer);instance._fast_tokenizer=self.fast;instance._fast_skip_tokens=128
        valid=self.fast(np.zeros((10,7)))[0]
        for ids in [valid,[309,1555,222,339,369,439,266,759,274,287,261,864]]:
            instance._paligemma_tokenizer=Paligemma(ids)
            actual=instance.extract_actions(np.array([1]),10,7)
            if ids==valid:
                self.assertTrue(last_decode['valid'])
                np.testing.assert_array_equal(actual,FASTTokenizer.extract_actions(instance,np.array([1]),10,7))
            else:
                self.assertFalse(last_decode['valid']);np.testing.assert_array_equal(actual,np.zeros((10,7)))
if __name__=='__main__':unittest.main()
