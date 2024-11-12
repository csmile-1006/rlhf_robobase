import logging
import uuid
from robobase.rlhf_module.utils import retry_on_error


class MockURI:
    def __init__(self, uri):
        self.uri = uri
        self.name = uuid.uuid4()

    def delete(self):
        pass


@retry_on_error(10)
def upload_video_from_genai_mock(video_path):
    mock_uri = f"mock_video_{uuid.uuid4()}.mp4"
    logging.info("Mocking upload video from GenAI")
    logging.info(f"Video processing complete: {mock_uri}")
    return MockURI(mock_uri)
