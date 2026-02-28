from PIL import Image
import os
import tqdm

def convert_tif_to_png(source_dir, target_dir):
    """
    Converts all .tif files in source_dir to .png files in target_dir.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.tif', '.tiff'))]
    print(f"Found {len(files)} TIF files.")

    for filename in tqdm.tqdm(files, desc="Converting"):
        file_path = os.path.join(source_dir, filename)
        name_no_ext = os.path.splitext(filename)[0]
        target_path = os.path.join(target_dir, name_no_ext + ".png")

        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (e.g. if CMYK or Grayscale with Alpha)
                # Usually material images are Grayscale (L) or RGB.
                # Direct save usually works, but converting to RGB ensures compatibility if it's weird 16-bit or something.
                # However, for scientific data, keeping original mode is best if possible.
                # PNG supports 8-bit grayscale (L) and RGB.
                # If it's floating point TIFF, that's trickier, but usually these are 8-bit or 16-bit int.
                # Let's try direct save.
                img.save(target_path, "PNG")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

    print("Conversion complete!")

if __name__ == "__main__":
    # Source: The "单晶图像" folder on user's desktop
    source_folder = r"c:\Users\pyd111\Desktop\标注2\单晶图像"
    
    # Target: Create a new folder "converted_pngs" inside the source folder
    # or just outside? Inside is cleaner for now as a subfolder, 
    # OR better yet, a parallel folder "单晶图像_png" to avoid recursion issues if we run it again
    # Let's put it in "c:\Users\pyd111\Desktop\标注2\单晶图像_png"
    target_folder = r"c:\Users\pyd111\Desktop\标注2\单晶图像_png"

    convert_tif_to_png(source_folder, target_folder)
