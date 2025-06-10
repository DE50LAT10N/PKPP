from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, color, shape, text=None):
    # Create a new image with a transparent background
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Calculate positions
    margin = size // 8
    inner_size = size - 2 * margin
    
    if shape == 'folder':
        # Draw folder shape
        draw.rectangle([margin, margin, size - margin, size - margin], 
                     fill=color, outline=(0, 0, 0, 255), width=2)
        draw.rectangle([margin, margin, size - margin, margin + inner_size//3], 
                     fill=color, outline=(0, 0, 0, 255), width=2)
    elif shape == 'process':
        # Draw circular arrow
        center = size // 2
        radius = inner_size // 2
        draw.arc([center - radius, center - radius, center + radius, center + radius],
                0, 270, fill=color, width=3)
        # Draw arrow head
        arrow_size = size // 8
        draw.polygon([
            (center + radius - arrow_size, center),
            (center + radius, center - arrow_size),
            (center + radius, center + arrow_size)
        ], fill=color)
    elif shape == 'generate':
        # Draw magic wand
        wand_length = inner_size
        wand_width = size // 8
        draw.rectangle([center - wand_width//2, margin, 
                       center + wand_width//2, margin + wand_length],
                      fill=color)
        # Draw star
        star_size = size // 4
        points = []
        for i in range(5):
            angle = i * 72 - 90
            x = center + int(star_size * 0.5 * (1 if i % 2 == 0 else 0.3) * 
                           (1 if i % 2 == 0 else -1) * (1 if i < 2 else -1))
            y = margin + int(star_size * 0.5 * (1 if i % 2 == 0 else 0.3) * 
                           (1 if i % 2 == 0 else -1) * (1 if i < 2 else -1))
            points.append((x, y))
        draw.polygon(points, fill=color)
    elif shape == 'export':
        # Draw download arrow
        arrow_size = inner_size // 2
        draw.polygon([
            (center, margin + arrow_size),
            (center - arrow_size//2, margin),
            (center + arrow_size//2, margin)
        ], fill=color)
        draw.rectangle([center - arrow_size//4, margin + arrow_size//2,
                       center + arrow_size//4, size - margin],
                      fill=color)
    
    # Add text if provided
    if text:
        try:
            font = ImageFont.truetype("arial.ttf", size//4)
        except:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.text((center - text_width//2, size - margin - text_height),
                 text, fill=(0, 0, 0, 255), font=font)
    
    return image

def create_loading_gif():
    # Create a series of frames for the loading animation
    frames = []
    size = 64
    num_frames = 12
    
    for i in range(num_frames):
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw rotating circle
        center = size // 2
        radius = size // 4
        start_angle = i * (360 / num_frames)
        end_angle = start_angle + 270
        
        draw.arc([center - radius, center - radius, center + radius, center + radius],
                start_angle, end_angle, fill=(52, 152, 219, 255), width=3)
        
        frames.append(image)
    
    # Save as GIF
    frames[0].save('loading.gif', save_all=True, append_images=frames[1:],
                  duration=100, loop=0, transparency=0)

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('icons'):
        os.makedirs('icons')
    
    # Define icon properties
    icons = [
        ('folder.png', (52, 152, 219, 255), 'folder'),
        ('process.png', (46, 204, 113, 255), 'process'),
        ('generate.png', (155, 89, 182, 255), 'generate'),
        ('export.png', (230, 126, 34, 255), 'export')
    ]
    
    # Generate icons
    for filename, color, shape in icons:
        icon = create_icon(64, color, shape)
        icon.save(os.path.join('icons', filename))
    
    # Generate loading animation
    create_loading_gif()
    
    print("Icons generated successfully!")

if __name__ == "__main__":
    main() 