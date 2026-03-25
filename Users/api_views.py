import os
import uuid
import json
import numpy as np
import cv2
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ValidationError
# import load_model moved to api_prediction function
# import tensorflow as tf moved to functions
from .models import userRegisteredTable


@csrf_exempt
def api_register(request):
    """API endpoint for user registration from mobile app."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name', '')
            email = data.get('email', '')
            loginid = data.get('loginid', '')
            mobile = data.get('mobile', '')
            password = data.get('password', '')

            # Check if loginid or email already exists
            if userRegisteredTable.objects.filter(loginid=loginid).exists():
                return JsonResponse({'status': 'error', 'message': 'Username already exists.'}, status=400)
            if userRegisteredTable.objects.filter(email=email).exists():
                return JsonResponse({'status': 'error', 'message': 'Email already registered.'}, status=400)

            user = userRegisteredTable(
                name=name,
                email=email,
                loginid=loginid,
                mobile=mobile,
                password=password,
            )
            user.full_clean()
            user.save()

            return JsonResponse({
                'status': 'success',
                'message': 'Registration successful! Please wait for admin approval.'
            })
        except ValidationError as ve:
            errors = []
            for field, errs in ve.message_dict.items():
                for err in errs:
                    errors.append(f"{field}: {err}")
            return JsonResponse({'status': 'error', 'message': '; '.join(errors)}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON data.'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Only POST method allowed.'}, status=405)


@csrf_exempt
def api_login(request):
    """API endpoint for user login from mobile app."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            loginid = data.get('loginid', '')
            password = data.get('password', '')

            try:
                user = userRegisteredTable.objects.get(loginid=loginid, password=password)
            except userRegisteredTable.DoesNotExist:
                return JsonResponse({'status': 'error', 'message': 'Invalid username or password.'}, status=401)

            if user.status != 'Active':
                return JsonResponse({
                    'status': 'error',
                    'message': 'Account not activated. Please wait for admin approval.'
                }, status=403)

            return JsonResponse({
                'status': 'success',
                'message': 'Login successful!',
                'user': {
                    'id': user.id,
                    'name': user.name,
                    'email': user.email,
                    'loginid': user.loginid,
                }
            })
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON data.'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Only POST method allowed.'}, status=405)

# Custom metric (used during model load)
def foreground_accuracy(y_true, y_pred):
    import tensorflow as tf
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    mask = tf.cast(tf.reduce_max(y_true, axis=-1, keepdims=True) > 0, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_bin), tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)


def validate_ct_scan(img):
    """
    Validate whether the uploaded image looks like a CT scan.
    CT scans are grayscale images with specific intensity characteristics.
    Returns (is_valid, error_message).
    """
    # Check 1: Image should be predominantly grayscale
    # CT scans have very low color saturation
    if len(img.shape) == 3 and img.shape[2] == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        # CT scans have very low saturation (< 30 typically)
        # Colorful photos/images have high saturation (> 50)
        if mean_saturation > 50:
            return False, "Invalid image: This does not appear to be a CT scan. Please upload a valid liver CT scan image."

    # Check 2: Image should have a reasonable intensity distribution
    # CT scans typically have dark backgrounds with lighter tissue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Very bright images (like documents, white backgrounds) are not CT scans
    if mean_intensity > 220:
        return False, "Invalid image: This does not appear to be a CT scan. Please upload a valid liver CT scan image."

    # Images with very low contrast (flat color) are not CT scans
    if std_intensity < 10:
        return False, "Invalid image: This does not appear to be a CT scan. Please upload a valid liver CT scan image."

    return True, ""


def validate_prediction_mask(pred_mask):
    """
    Validate the prediction mask to check if a liver region was found.
    If the mask is nearly empty or nearly full, the image is likely not a valid liver CT scan.
    Returns (is_valid, error_message).
    """
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
    total_pixels = pred_mask_bin.size
    white_pixels = np.sum(pred_mask_bin)
    coverage = white_pixels / total_pixels

    # If less than 0.5% of pixels are detected as liver, no liver found
    if coverage < 0.005:
        return False, "Invalid image: No liver region detected. Please upload a valid liver CT scan image."

    # If more than 70% is detected, the prediction is unreliable (not a real CT scan)
    if coverage > 0.70:
        return False, "Invalid image: This does not appear to be a valid liver CT scan image."

    return True, ""


@csrf_exempt
def api_prediction(request):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    if request.method == 'POST' and 'image' in request.FILES:
        try:
            # Save uploaded image
            uploaded_file = request.FILES['image']
            unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', unique_filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            with open(image_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Load and preprocess image
            IMG_HEIGHT, IMG_WIDTH = 256, 256
            img = cv2.imread(image_path)
            if img is None:
                return JsonResponse({'status': 'error', 'message': 'Invalid image file.'}, status=400)

            # Step 1: Validate that the image looks like a CT scan
            is_valid, error_msg = validate_ct_scan(img)
            if not is_valid:
                return JsonResponse({'status': 'error', 'message': error_msg}, status=400)

            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            img_input = np.expand_dims(img_resized, axis=0)

            # Load model
            model_path = os.path.join(settings.MEDIA_ROOT, "liver_segmentation_unet.h5")
            if not os.path.exists(model_path):
                 return JsonResponse({'status': 'error', 'message': 'Model file not found.'}, status=500)
                 
            model = load_model(model_path, custom_objects={"foreground_accuracy": foreground_accuracy})

            # Predict
            pred_mask = model.predict(img_input)[0]

            # Step 2: Validate the prediction mask (check if liver was found)
            is_valid, error_msg = validate_prediction_mask(pred_mask)
            if not is_valid:
                return JsonResponse({'status': 'error', 'message': error_msg}, status=400)

            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

            # Save predicted mask
            mask_filename = f"predicted_{unique_filename}"
            mask_path = os.path.join(settings.MEDIA_ROOT, 'masks', mask_filename)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, pred_mask_bin.squeeze())

            # Static URL base
            base_url = request.build_absolute_uri(settings.MEDIA_URL)
            
            return JsonResponse({
                'status': 'success',
                'image_url': f"{base_url}uploads/{unique_filename}",
                'mask_url': f"{base_url}masks/{mask_filename}"
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request.'}, status=400)
