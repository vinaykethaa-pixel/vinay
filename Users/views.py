import shutil
from django.shortcuts import render
from sklearn.model_selection import train_test_split







from .models import userRegisteredTable
from django.core.exceptions import ValidationError
from django.contrib import messages


def userRegisterCheck(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        username = request.POST.get("loginId")
        mobile = request.POST.get("mobile")
        password = request.POST.get("password")
        

        # Create an instance of the model
        user = userRegisteredTable(
            name=name,
            email=email,
            loginid=username,
            mobile=mobile,
            password=password,
            
        )

        try:
            # Validate using model field validators
            user.full_clean()
            
            # Save to DB
            user.save()
            messages.success(request,'registration Successfully done,please wait for admin APPROVAL')
            return render(request, "userRegisterForm.html")


        except ValidationError as ve:
            # Get a list of error messages to display
            error_messages = []
            for field, errors in ve.message_dict.items():
                for error in errors:
                    error_messages.append(f"{field.capitalize()}: {error}")
            return render(request, "userRegisterForm.html", {"messages": error_messages})

        except Exception as e:
            # Handle other exceptions (like unique constraint fails)
            return render(request, "userRegisterForm.html", {"messages": [str(e)]})

    return render(request, "userRegisterForm.html")


def userLoginCheck(request):
    if request.method=='POST':
        username=request.POST['userUsername']
        password=request.POST['userPassword']

        try:
            user=userRegisteredTable.objects.get(loginid=username,password=password)

            if user.status=='Active':
                request.session['id']=user.id
                request.session['name']=user.name
                request.session['email']=user.email
                
                return render(request,'users/userHome.html')
            else:
                messages.error(request,'Status not activated please wait for admin approval')
                return render(request,'userLoginForm.html')
        except:
            messages.error(request,'Invalid details please enter details carefully or Please Register')
            return render(request,'userLoginForm.html')
    return render(request,'userLoginForm.html')


def userHome(request):
    if not request.session.get('id'):
        return render(request,'userLoginForm.html')
    return render(request,'users/userHome.html')
def Ulog(request):
    request.session.flush()  # clears all session data
    return render(request,'userLoginForm.html')
import os
import numpy as np
# import tensorflow as tf moved to functions
# from tensorflow.keras import layers, models moved to functions
import cv2
import pandas as pd


# # Settings
# train_dir = r"/content/extracted_files/div-images/train"
# test_dir = r"/content/extracted_files/div-images/test"
# IMG_HEIGHT, IMG_WIDTH = 256, 256
# BATCH_SIZE = 8
# EPOCHS = 20

# DataGen class moved inside training function to delay tensorflow loading

# Foreground accuracy metric
def foreground_accuracy(y_true, y_pred):
    import tensorflow as tf
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    mask = tf.cast(tf.reduce_max(y_true, axis=-1, keepdims=True) > 0, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_bin), tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)

# Simple U-Net model
def build_unet(input_shape=(256, 256, 3)):
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    return models.Model(inputs, outputs)
def training(request):
    import tensorflow as tf
    # Data generator for files like 070.png and 070_mask.png
    class DataGen(tf.keras.utils.Sequence):
        def __init__(self, folder, batch_size, img_size):
            self.folder = folder
            self.batch_size = batch_size
            self.img_size = img_size
            self.image_files = sorted([
                f for f in os.listdir(folder)
                if f.endswith(".png") and not "_mask" in f
            ])

        def __len__(self):
            return len(self.image_files) // self.batch_size

        def __getitem__(self, idx):
            batch_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
            images, masks = [], []
            for file in batch_files:
                base_name = file.replace(".png", "")
                img_path = os.path.join(self.folder, f"{base_name}.png")
                mask_path = os.path.join(self.folder, f"{base_name}_mask.png")

                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, self.img_size) / 255.0
                mask = cv2.resize(mask, self.img_size)
                mask = (mask > 127).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)

                images.append(img)
                masks.append(mask)

            return np.array(images), np.array(masks)
    
    if not request.session.get('id'):
        return render(request,'userLoginForm.html')
        
    # # Compile model
    # model = build_unet()
    # model.compile(optimizer='adam',
    #             loss='binary_crossentropy',
    #             metrics=['accuracy', foreground_accuracy])

    # # Generators
    # train_gen = DataGen(train_dir, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))
    # val_gen = DataGen(test_dir, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))

    # # Logging
    # csv_logger = tf.keras.callbacks.CSVLogger("training_log.csv", append=False)

    # # Train
    # history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[csv_logger])

    # # Save model and training history
    # model.save("media/liver_segmentation_unet.h5")
    # pd.DataFrame(history.history).to_csv("media/metrics_summary.csv", index=False)
    results1=pd.read_csv('media/metrics_summary.csv')
    dff=results1.to_html()
    # Pass DataFrame to template (convert to dict for easier rendering)
    return render(request, 'users/training.html', {
         
        'results_df':dff  # Convert DataFrame to list of dictionaries
    })


import os
import uuid
import numpy as np
import cv2
from django.conf import settings
from django.shortcuts import render
# import load_model moved to prediction function
# import tensorflow as tf moved to prediction function

# Custom metric (used during model load)
def foreground_accuracy(y_true, y_pred):
    import tensorflow as tf
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    mask = tf.cast(tf.reduce_max(y_true, axis=-1, keepdims=True) > 0, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_bin), tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)

def prediction(request):
    if not request.session.get('id'):
        return render(request, 'userLoginForm.html')

    if request.method == 'POST' and 'image' in request.FILES:
        import gc
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Optimize TensorFlow for Render's 512MB RAM constraint
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
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
                return render(request, 'users/prediction.html', {
                    'error': 'Invalid image file.'
                })

            # Step 1: Validate that the image looks like a CT scan
            # Check grayscale (low saturation)
            is_ct_valid = True
            error_msg = None

            if len(img.shape) == 3 and img.shape[2] == 3:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mean_saturation = np.mean(hsv[:, :, 1])
                if mean_saturation > 50:
                    is_ct_valid = False
                    error_msg = "Invalid image: This does not appear to be a CT scan. Please upload a valid liver CT scan image."

            if is_ct_valid:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                mean_intensity = np.mean(gray)
                std_intensity = np.std(gray)
                if mean_intensity > 220:
                    is_ct_valid = False
                    error_msg = "Invalid image: This does not appear to be a CT scan. Please upload a valid liver CT scan image."
                elif std_intensity < 10:
                    is_ct_valid = False
                    error_msg = "Invalid image: This does not appear to be a CT scan. Please upload a valid liver CT scan image."

            if not is_ct_valid:
                return render(request, 'users/prediction.html', {
                    'error': error_msg
                })

            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
            img_input = np.expand_dims(img_resized, axis=0)

            # Load model
            model_path = os.path.join(settings.MEDIA_ROOT, "liver_segmentation_unet.h5")

            # Patch the model config to fix "batch_shape" -> "batch_input_shape" for Keras 3 / TF 2.15+ compatibility
            try:
                import h5py
                import json
                with h5py.File(model_path, 'r+') as f:
                    model_config_str = f.attrs.get('model_config')
                    if model_config_str:
                        if isinstance(model_config_str, bytes):
                            model_config_str = model_config_str.decode('utf-8')
                        model_config = json.loads(model_config_str)
                        changed = False
                        for layer in model_config.get('config', {}).get('layers', []):
                            if layer.get('class_name') == 'InputLayer':
                                if 'batch_shape' in layer.get('config', {}):
                                    layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                                    changed = True
                            
                            # Patch Keras 3 DTypePolicy back to Keras 2 string dtype format
                            if 'dtype' in layer.get('config', {}):
                                dtype_val = layer['config']['dtype']
                                if isinstance(dtype_val, dict) and dtype_val.get('class_name') == 'DTypePolicy':
                                    layer['config']['dtype'] = dtype_val.get('config', {}).get('name', 'float32')
                                    changed = True
                        if changed:
                            f.attrs.modify('model_config', json.dumps(model_config).encode('utf-8'))
            except Exception as e:
                print(f"Issue patching h5 file (could be already patched or read-only): {e}")

            model = load_model(model_path, custom_objects={"foreground_accuracy": foreground_accuracy})

            # Predict
            pred_mask = model.predict(img_input)[0]

            # Step 2: Validate the prediction mask (check if liver was found)
            pred_mask_check = (pred_mask > 0.5).astype(np.uint8)
            coverage = np.sum(pred_mask_check) / pred_mask_check.size

            if coverage < 0.005:
                return render(request, 'users/prediction.html', {
                    'error': 'Invalid image: No liver region detected. Please upload a valid liver CT scan image.'
                })
            if coverage > 0.70:
                return render(request, 'users/prediction.html', {
                    'error': 'Invalid image: This does not appear to be a valid liver CT scan image.'
                })

            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

            # Save predicted mask
            mask_filename = f"predicted_{unique_filename}"
            mask_path = os.path.join(settings.MEDIA_ROOT, 'masks', mask_filename)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, pred_mask_bin.squeeze())

            # URLs to display in template
            image_url = os.path.join(settings.MEDIA_URL, 'uploads', unique_filename)
            mask_url = os.path.join(settings.MEDIA_URL, 'masks', mask_filename)

            return render(request, 'users/prediction.html', {
                'predicted_class': 'Liver Region',
                'confidence': None,
                'image_url': image_url,
                'mask_url': mask_url,
                'error': None
            })
        finally:
            tf.keras.backend.clear_session()
            gc.collect()

    return render(request, 'users/prediction.html', {
        'predicted_class': None,
        'confidence': None,
        'image_url': None,
        'mask_url': None,
        'error': 'No image uploaded or invalid request method.'
    })
