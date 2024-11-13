from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('damo/ofa_mmspeech_asr_aishell1_base_zh', cache_dir='MmspeechModelWeight')
model_dir = snapshot_download('damo/ofa_mmspeech_asr_aishell1_large_zh', cache_dir='MmspeechModelWeight')
model_dir = snapshot_download('damo/ofa_mmspeech_pretrain_base_zh', cache_dir='MmspeechModelWeight')
model_dir = snapshot_download('damo/ofa_mmspeech_pretrain_large_zh', cache_dir='MmspeechModelWeight')