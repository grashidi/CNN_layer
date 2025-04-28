import ctypes
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

lib = ctypes.CDLL('./build/libcnn_shared.so')

lib.run_conv_2d_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # image
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # result
    ctypes.c_uint32,  # width
    ctypes.c_uint32,  # height
    ctypes.c_uint32,  # pool_w
    ctypes.c_uint32,  # pool_h
    ctypes.POINTER(ctypes.c_float),  # kernelX
    ctypes.POINTER(ctypes.c_float),  # kernelY
    ctypes.c_int  # kernel_size
]

lib.run_conv_2d_transpose_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # image
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # result
    ctypes.c_uint32,  # width
    ctypes.c_uint32,  # height
    ctypes.POINTER(ctypes.c_float),  # kernelX
    ctypes.POINTER(ctypes.c_float),  # kernelY
    ctypes.c_int  # kernel_size
]

def cuda_encode(image, kernelX, kernelY, width, height, pool_w, pool_h, kernel_size):
    input_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    temp_buffer = (ctypes.c_ubyte * ((height // pool_h) * (width // pool_w)))()
    output_ptr = ctypes.cast(temp_buffer, ctypes.POINTER(ctypes.c_ubyte))

    lib.run_conv_2d_kernel(
        input_ptr,
        ctypes.byref(output_ptr),
        ctypes.c_uint32(width),
        ctypes.c_uint32(height),
        ctypes.c_uint32(pool_w),
        ctypes.c_uint32(pool_h),
        kernelX.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kernelY.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(kernel_size)  # assuming kernel is square
    )
    return output_ptr

def cuda_decode(encoded, kernelX, kernelY, enc_w, enc_h, kernel_size):
    out_h = (enc_h - 1) * 2 + kernel_size
    out_w = (enc_w - 1) * 2 + kernel_size
    result = np.zeros((out_h * out_w), dtype=np.float32)
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    lib.run_conv_2d_transpose_kernel(
        encoded,
        result_ptr,
        ctypes.c_uint32(enc_w),
        ctypes.c_uint32(enc_h),
        kernelX.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kernelY.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(kernel_size)
    )

    arr = np.ctypeslib.as_array(result_ptr, shape=(out_h * out_w,))
    return arr.reshape((out_h, out_w))

def loss_fn(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

learning_rate = 1e-2
epsilon = 1e-3
kernel_size = 3
kernelX = np.random.randn(9).astype(np.float32)
kernelY = np.random.randn(9).astype(np.float32)

m_X, v_X = np.zeros_like(kernelX), np.zeros_like(kernelX)
m_Y, v_Y = np.zeros_like(kernelY), np.zeros_like(kernelY)
beta1, beta2 = 0.9, 0.999
epsilon_adam = 1e-8
t = 0

images_path = glob.glob("images/aerials_original/*.tiff")
training_images = [np.array(Image.open(f).convert("L"), dtype=np.uint8) for f in images_path]

# Before training, print initial kernels
print("Initial kernelX:", kernelX)
print("Initial kernelY:", kernelY)

for epoch in range(10):
    total_loss = 0.0
    max_grad_x, max_grad_y = 0, 0

    progress_bar = tqdm(training_images, desc=f"Epoch {epoch}", leave=False)
    for img_idx, img in enumerate(progress_bar):
        h, w = img.shape
        enc_w, enc_h = w // 2, h // 2

        # Forward pass
        encoded = cuda_encode(img.flatten(), kernelX, kernelY, w, h, 2, 2, kernel_size)
        decoded = cuda_decode(encoded, kernelX, kernelY, enc_w, enc_h, kernel_size)
        loss = loss_fn(img, decoded[:h, :w])
        total_loss += loss

        # Compute gradients with temporary kernel copies
        gradX = np.zeros_like(kernelX)
        gradY = np.zeros_like(kernelY)

        for i in range(kernel_size * kernel_size):
            # For kernelX
            temp_kernelX = kernelX.copy()
            temp_kernelX[i] += epsilon
            enc_eps = cuda_encode(img.flatten(), temp_kernelX, kernelY, w, h, 2, 2, kernel_size)
            dec_eps = cuda_decode(enc_eps, temp_kernelX, kernelY, enc_w, enc_h, kernel_size)
            loss_eps = loss_fn(img, dec_eps[:h, :w])
            gradX[i] = (loss_eps - loss) / epsilon

            # For kernelY
            temp_kernelY = kernelY.copy()
            temp_kernelY[i] += epsilon
            enc_eps = cuda_encode(img.flatten(), kernelX, temp_kernelY, w, h, 2, 2, kernel_size)
            dec_eps = cuda_decode(enc_eps, kernelX, temp_kernelY, enc_w, enc_h, kernel_size)
            loss_eps = loss_fn(img, dec_eps[:h, :w])
            gradY[i] = (loss_eps - loss) / epsilon

        # Track maximum gradients
        max_grad_x = max(max_grad_x, np.max(np.abs(gradX)))
        max_grad_y = max(max_grad_y, np.max(np.abs(gradY)))

        # Gradient descent step
        kernelX_before = kernelX.copy()
        kernelY_before = kernelY.copy()

        t += 1

        m_X = beta1 * m_X + (1 - beta1) * gradX
        v_X = beta2 * v_X + (1 - beta2) * np.square(gradX)
        m_X_hat = m_X / (1 - beta1**t)
        v_X_hat = v_X / (1 - beta2**t)
        kernelX -= learning_rate * m_X_hat / (np.sqrt(v_X_hat) + epsilon_adam)

        # Similarly for kernelY
        m_Y = beta1 * m_Y + (1 - beta1) * gradY
        v_Y = beta2 * v_Y + (1 - beta2) * np.square(gradY)
        m_Y_hat = m_Y / (1 - beta1**t)
        v_Y_hat = v_Y / (1 - beta2**t)
        kernelY -= learning_rate * m_Y_hat / (np.sqrt(v_Y_hat) + epsilon_adam)

    avg_loss = total_loss / len(training_images)
    tqdm.write(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f}")

    img = training_images[0]
    h, w = img.shape
    enc_w, enc_h = w // 2, h // 2
    img = img.flatten()

    enc_img = cuda_encode(img.flatten(), kernelX, kernelY, w, h, 2, 2, kernel_size)
    result = np.ctypeslib.as_array(
        enc_img, shape=(enc_h, enc_w)
    ).copy()

    # Save or display result
    Image.fromarray(result).save("images/processed/" + "epoch_" + str(epoch) + "_" + images_path[0].split("/")[-1])