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
