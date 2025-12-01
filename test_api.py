import urllib.request
import urllib.parse
import json
import os
import mimetypes

BASE_URL = "http://127.0.0.1:8000"

def post_multipart(url, fields, files):
    boundary = '----------ThIs_Is_tHe_bouNdaRY_$'
    data = []
    for (key, value) in fields.items():
        data.append('--' + boundary)
        data.append('Content-Disposition: form-data; name="%s"' % key)
        data.append('')
        data.append(value)
    
    for (key, filename) in files.items():
        content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        with open(filename, 'rb') as f:
            file_content = f.read()
        data.append('--' + boundary)
        data.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, os.path.basename(filename)))
        data.append('Content-Type: %s' % content_type)
        data.append('')
        data.append(file_content.decode('latin-1')) # Hacky but works for bytes in this simple case

    data.append('--' + boundary + '--')
    data.append('')
    body = '\r\n'.join(data).encode('latin-1')
    
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

def test_registration(image_path, pension_id, name):
    print(f"Testing Registration for {name}...")
    status, response = post_multipart(f"{BASE_URL}/register", {"pension_id": pension_id, "name": name}, {"file": image_path})
    print(f"Status: {status}")
    print(f"Response: {response}")
    return status == 200

def test_verification(image_path, pension_id):
    print(f"Testing Verification for {pension_id}...")
    status, response = post_multipart(f"{BASE_URL}/verify", {"pension_id": pension_id}, {"file": image_path})
    print(f"Status: {status}")
    print(f"Response: {response}")
    return response

if __name__ == "__main__":
    if not os.path.exists("test_face.jpg"):
        print("WARNING: No test image found. Tests will fail.")
    else:
        # 1. Register
        test_registration("test_face.jpg", "12345", "John Doe")
        
        # 2. Verify Success (Same image)
        test_verification("test_face.jpg", "12345")
        
        # 3. Verify Fail (Wrong ID)
        test_verification("test_face.jpg", "99999")
