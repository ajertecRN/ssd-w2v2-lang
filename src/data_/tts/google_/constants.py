from google.cloud.texttospeech import AudioEncoding as AE

GOOGLE_AUDIO_ENCODING_STRING_TO_ENUM = {
    "alaw": AE.ALAW,
    "linear16": AE.LINEAR16,
    "mp3": AE.MP3,
    "mulaw": AE.MULAW,
    "ogg_opus": AE.OGG_OPUS,
}

GOOGLE_AUDIO_ENCODING_STRING_TO_EXTENSION = {
    "alaw": "wav",
    "linear16": "wav",
    "mp3": "mp3",
    "mulaw": "wav",
    "ogg_opus": "opus",
}
