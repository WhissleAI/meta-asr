from dotenv import load_dotenv
from pyannote.audio import Pipeline
import os
load_dotenv()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=os.getenv("hf_toekn") )


# apply the pipeline to an audio file
diarization = pipeline("test_audio/test2.mp3")

# dump the diarization output to disk using RTTM format
with open("output/audio1.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
