# IDP Boot Camp – Week 3 Assignment – Table Extraction

## 1. Table extraction

This project aims to perform table detection using Microsoft's Table Transformer model from Hugging Face.

### Prerequisites

Make sure you have Python 3.6 or higher installed on your machine.

### Installation

Install the required libraries:

```bash
# Install transformers for loading pre-trained models
%pip install transformers

# Install pillow for image processing
%pip install pillow

# Install huggingface_hub for accessing models and datasets
%pip install huggingface_hub

# Install timm for the model architecture
%pip install timm

# Install matplotlib for visualization
%pip install matplotlib
```

### draw the bounding box in detected the table and save

After detecting the table, we draw a bounding box around the table in the image and save it
saved image:![drawed bounding box](/Table%20detection_images/rec_000f8315c3bea30fb7ae99f925286343-15.png)

## Table structure recognition

Model Loading: The model is loaded from Hugging Face's model hub, ensuring you're using the latest version.
Feature Extraction: The DetrFeatureExtractor is used to prepare the input table image for the model.
Bounding Box Extraction: The code extracts bounding boxes and scores for rows, columns, and column headers based on the labels defined in the model configuration.
Output: The function returns bounding boxes and their corresponding scores for rows, columns, and headers, which can be used for further processing or visualization.

```python
#get the model
model_structure = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition")


def get_row_col_bounds(table, ts_thresh=0.7, plot=False):
    feature_extractor = DetrFeatureExtractor()
    table_encoding = feature_extractor(table, return_tensors="pt")

    # predict table structure
    with torch.no_grad():
        outputs = model_structure(**table_encoding)

    # visualize table structure
    target_sizes = [table.size[::-1]]
    table_struct_results = feature_extractor.post_process_object_detection(
        outputs, threshold=ts_thresh, target_sizes=target_sizes
    )[0]


    row_boxes = table_struct_results["boxes"][
        table_struct_results["labels"] == model_structure.config.label2id["table row"]
    ]

    row_scores = table_struct_results["scores"][
        table_struct_results["labels"] == model_structure.config.label2id["table row"]
    ]

    col_boxes = table_struct_results["boxes"][
        table_struct_results["labels"]
        == model_structure.config.label2id["table column"]
    ]

    col_scores = table_struct_results["scores"][
        table_struct_results["labels"]
        == model_structure.config.label2id["table column"]
    ]

    table_header_box = table_struct_results["boxes"][
        table_struct_results["labels"]
        == model_structure.config.label2id["table column header"]
    ]
    table_header_score = table_struct_results["scores"][
        table_struct_results["labels"]
        == model_structure.config.label2id["table column header"]
    ]

    print(f"Num rows initially detected: {len(row_boxes)}")
    print(f"Num cols initially detected: {len(col_boxes)}")
    print(f"Num table header detected: {len(table_header_box)}")


    return (
        row_boxes,
        row_scores,
        col_boxes,
        col_scores,
        table_header_box,
        table_header_score,
    )

```

after structure recognition,drawd the row structure we saved the images
saved image:![drawed table structures](/Table%20structure%20recognition_image/structure_0005ab896e95c5de6885bbf595500a8e-66.png)

add the padding to the image and crop the image and save that in to folder

```python
for i in range(len(image_paths)):
  padding = 15
  box=total_box[i]
  image=Image.open(image_paths[i]).convert("RGB")
  box = [box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding]
  table_image = image.crop(box)
  save_path='/content/drive/MyDrive/idp_bootcamp/week_3_assignment/Table structure recognition_image/'+'structure_'+image_name1[i]
  output_img=plot_box(table_image, total_table_structure_outs[i][0])
  output_img.save(save_path)
```

## Decomposing cells

Initialization:

Three lists are created to store detected cells, sorted row boundaries, and sorted column boundaries for multiple images.
Image Processing Loop:

The code loops through a list of image paths, processing each image one by one.
Bounding Box Adjustment:

For each image, a bounding box is retrieved, and padding is added to the box to create a larger area for cropping.
Image Cropping and Padding:

The image is opened, cropped to the adjusted bounding box, and additional padding is applied before saving it.
Row and Column Boundary Detection:

The padded image is analyzed to detect the boundaries of rows and columns within the table.
Visualization:

Detected boundaries are plotted on the padded image for visual confirmation.
Sorting and Cell Extraction:

The row and column boundaries are sorted, and cells are extracted by intersecting the sorted rows and columns.
Results Storage:

The detected cells, sorted row boundaries, and sorted column boundaries are appended to their respective lists for each processed image.

Overall, the code processes a series of images containing tables, extracting relevant structural information (cells, rows, and columns) while visualizing the results and saving intermediate padded images.
visualization ![fully visulaize table structures](</download%20(1).png>)

## change table to CSV file

we need to download the paddle ocr to detect the text in the each cell

```base
%pip install paddlepaddle paddleocr
%pip install opencv-python
```

1.Initialization:
An empty list, over_results, is created to store the OCR results for each image processed.

2.Outer Loop:
Iterates over the list of padded image names (image_name_padded).

3.Image Handling:
For each image
Constructs the file path of the padded image.
Opens the image and converts it to RGB format.

4.Inner Loop:
For each cell (bounding box) in the corresponding list
Crops the image to the current cell's bounding box.
Performs OCR on the cropped image using the ocr method from the PaddleOCR library.
Appends the OCR results for the current cell to the results list.

5.Result Storage:
After processing all cells for the current image, the results are appended to the over_results list.

example
![extracted data.csv](/csv_outs/0005ab896e95c5de6885bbf595500a8e-66.csv)
