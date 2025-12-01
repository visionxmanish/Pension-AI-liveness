import time
import uuid

def generate_meet_link():
    """Generates a unique Jitsi Meet link."""
    # Using a unique identifier to ensure the room is private/unique
    unique_id = uuid.uuid4().hex[:10]
    timestamp = int(time.time())
    return f"https://meet.jit.si/PensionVerification_{unique_id}_{timestamp}"
