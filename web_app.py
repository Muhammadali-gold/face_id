from fastapi import FastAPI, Form, UploadFile, File
import base64
from pydantic import BaseModel
from uuid import UUID
import uvicorn
from main import has_multiple_face, get_similarity
app = FastAPI()

class ImageRequest(BaseModel):
    client_id: UUID
    real_image: str  # base64 string
    target_image: str  # base64 string


@app.post("/v1/image/face/similar")
def read_root(request : ImageRequest):
    real_bytes = base64.b64decode(request.real_image)
    target_bytes = base64.b64decode(request.target_image)

    with open(f"{request.client_id}_real.png", "wb") as f:
        f.write(real_bytes)

    with open(f"{request.client_id}_target.png", "wb") as f:
        f.write(target_bytes)

    file_1 = f"{request.client_id}_real.png"
    file_2 = f"{request.client_id}_target.png"

    if has_multiple_face(file_1):
        return {"status": "error", "message": "First image has multiple faces."}
    elif has_multiple_face(file_2):
        return {"status": "error", "message": "Second image has multiple faces."}
    else:
        try:
            score = get_similarity(file_1, file_2)
            if score >= 0.7:
                return {"status": "success", "client_id": str(request.client_id),"result": f"Same person / similar ({score})", "score": score}
            else:
                return {"status": "success", "client_id": str(request.client_id),"result": f"Different person / similar ({score})", "score": score}
        except ValueError as e:
            return {"status": "error", "message": str(e)}


@app.post("/v2/image/face/similar")
def face_similarity_v2(
    client_id: str = Form(...),
    real_image: UploadFile = File(...),
    target_image: UploadFile = File(...)
):
    # Validate file types
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/jfif"]
    if real_image.content_type not in allowed_types:
        return {"status": "error", "message": f"Real image type {real_image.content_type} not supported. Allowed: {allowed_types}"}
    if target_image.content_type not in allowed_types:
        return {"status": "error", "message": f"Target image type {target_image.content_type} not supported. Allowed: {allowed_types}"}
    
    # Save uploaded files
    real_file_path = f"{client_id}_real.png"
    target_file_path = f"{client_id}_target.png"
    
    try:
        # Write real image to file
        with open(real_file_path, "wb") as f:
            f.write(real_image.file.read())
        
        # Write target image to file
        with open(target_file_path, "wb") as f:
            f.write(target_image.file.read())
        
        # Check for multiple faces
        if has_multiple_face(real_file_path):
            return {"status": "error", "message": "Real image has multiple faces."}
        elif has_multiple_face(target_file_path):
            return {"status": "error", "message": "Target image has multiple faces."}
        else:
            try:
                score = get_similarity(real_file_path, target_file_path)
                if score >= 0.7:
                    return {
                        "status": "success", 
                        "client_id": client_id,
                        "result": f"Same person / similar ({score})", 
                        "score": score
                    }
                else:
                    return {
                        "status": "success", 
                        "client_id": client_id,
                        "result": f"Different person / similar ({score})", 
                        "score": score
                    }
            except ValueError as e:
                return {"status": "error", "message": str(e)}
    
    except Exception as e:
        return {"status": "error", "message": f"Error processing files: {str(e)}"}


if __name__ == '__main__':
    uvicorn.run(
            "web_app:app",  # "filename:app_instance"
            host="0.0.0.0",
            port=8000,
            reload=True
        )
