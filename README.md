# Pension App Face Recognition API

This is a FastAPI-based backend for a pension application that handles face registration, recognition, and liveness detection (via quality checks).

## Features
- **Face Registration**: Store user details and face embeddings.
- **Face Verification**: Authenticate users using face recognition (DeepFace/ArcFace).
- **Liveness Detection**:
    - **Anti-Spoofing**: Checks if the face is real (using DeepFace's anti-spoofing model).
    - **Quality Checks**: Anti-glare and blur detection.
- **Fallback**: Generates a Jitsi Meet link if verification fails.

## Setup

1.  **Prerequisites**: Python 3.8+
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `deepface` and `opencv-python` are required.*

## Running the Application

1.  Start the server:
    ```bash
    uvicorn main:app --reload
    ```
2.  The API will be running at `http://127.0.0.1:8000`.
3.  Interactive Documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

## API Endpoints

### 1. Register User
- **URL**: `/register`
- **Method**: `POST`
- **Body** (`multipart/form-data`):
    - `pension_id` (string): Unique ID for the pensioner.
    - `name` (string): Name of the pensioner.
    - `file` (file): Face image (JPG/PNG).

### 2. Verify User
- **URL**: `/verify`
- **Method**: `POST`
- **Body** (`multipart/form-data`):
    - `pension_id` (string): Unique ID to verify.
    - `file` (file): New face image to compare.

### 3. Verify Liveness (Video)
- **URL**: `/verify_liveness_video`
- **Method**: `POST`
- **Body** (`multipart/form-data`):
    - `pension_id` (string): Unique ID to verify.
    - `verification_type` (string): Actions to check, comma-separated. E.g., `smile,move_head_left,move_head_right`.
    - `file` (file): Video file (`.mp4`) containing the user performing the actions.

## How to Test with Postman

Follow these steps to test the API using Postman:

### Step 1: Register a User
1.  Open Postman and create a new request.
2.  Set the method to **POST**.
3.  Set the URL to `http://127.0.0.1:8000/register`.
4.  Go to the **Body** tab.
5.  Select **form-data**.
6.  Add the following key-value pairs:
    - Key: `pension_id`, Value: `1001` (Type: Text)
    - Key: `name`, Value: `John Doe` (Type: Text)
    - Key: `file`, Value: [Select a clear face photo] (Type: File)
      *(Note: Hover over the "Value" field to see the file selector dropdown if it's not visible, or change the key type from "Text" to "File" using the dropdown on the right of the Key name)*
7.  Click **Send**.
8.  You should receive a JSON response with the user details.

### Step 2: Verify a User
1.  Create a new request.
2.  Set the method to **POST**.
3.  Set the URL to `http://127.0.0.1:8000/verify`.
4.  Go to the **Body** tab.
5.  Select **form-data**.
6.  Add the following key-value pairs:
    - Key: `pension_id`, Value: `1001` (Type: Text)
    - Key: `file`, Value: [Select a photo of the same person] (Type: File)
7.  Click **Send**.

## How to Test with OpenAPI (Swagger UI)

FastAPI automatically generates an interactive testing interface. This is the fastest way to test endpoints like the Video Liveness check.

1. Ensure the server is running (`uvicorn main:app --reload`).
2. Open your web browser and go to `http://127.0.0.1:8000/docs`.
3. You will see a list of available endpoints (`/register`, `/verify`, `/verify_liveness_video`).
4. Click on the endpoint you want to test (e.g., `POST /verify_liveness_video` to expand it).
5. Click the **"Try it out"** button in the top right corner of the expanded endpoint.
6. Fill in the required fields:
   - `pension_id`: Enter the registered ID (e.g., `1001`).
   - `verification_type`: Enter the actions you recorded in your video (e.g., `smile` or `move_head_left,turn_right`).
   - `file`: Click "Choose File" and upload your `.mp4` video.
7. Click the large blue **"Execute"** button below the form.
8. Scroll down slightly to see the **Server response** containing the success/failure JSON message.

### Interpreting Responses

- **Success**:
  ```json
  {
      "status": "success",
      "message": "Verification successful",
      "meet_link": null
  }
  ```

- **Failure (Mismatch or Glare)**:
  ```json
  {
      "status": "failure",
      "message": "Face does not match",
      "meet_link": "https://meet.jit.si/PensionVerification_..."
  }
  ```
  *If you see a "meet_link", the verification failed, and the app should redirect the user to a video call.*

## Troubleshooting
- **Glare Detected**: If you get a "Glare detected" error, try using a photo with more even lighting.
- **Blurry Image**: Ensure the photo is sharp.
- **Server Error**: Check the terminal output for detailed error logs.
