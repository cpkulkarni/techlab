import easyocr

# Define languages to recognize (English and Sanskrit)
# lang = ['sa']

# Create a Reader object
reader = easyocr.Reader(['mr'])

# Load the image, give full path if necessary
image_path = "test2.png"  

# get full details
# image = reader.readtext(image_path)

# get plain text
# image = reader.readtext(image_path, detail=0)

# get paragraph
image = reader.readtext(image_path, paragraph=True)

## Best to use paragraph option, so related text will be pt together

# print(image)

# encoding needss to be passe while storing data else encoding related errors will be raised
output_file = open("output.2.txt", "a", encoding="utf-8")
output_file.write(str(image))

# Process the results
# for text, bbox, prob in image:
#    print(f"Text: {text}, Bounding Box: {bbox}, Probability: {prob:.2f}")

    