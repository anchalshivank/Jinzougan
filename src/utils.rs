use std::cmp::max;
use log::error;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size},
    highgui, imgcodecs,
    imgproc::{
        self, bounding_rect, cvt_color, find_contours, threshold, ADAPTIVE_THRESH_MEAN_C,
        CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY, LINE_8, MORPH_OPEN, RETR_TREE, THRESH_BINARY,
    },
    prelude::*,
    types::VectorOfVectorOfPoint,
    Result,
};
use opencv::core::BORDER_DEFAULT;
use opencv::imgproc::rectangle;

const STAMP_HEIGHT: i32 = 500 / 3;
const STAMP_HEIGHT_LEFT_CORRECTION: i32 = 0;
const STAMP_HEIGHT_RIGHT_CORRECTION: i32 = 5;
const STAMP_WIDTH: i32 = 610;
const LEFT_OFFSET: i32 = 25;
const RIGHT_OFFSET: i32 = 105;

fn show_image(title: &str, image: &Mat) -> Result<()> {
    highgui::imshow(title, image)?;
    highgui::wait_key(0)?;
    Ok(())
}

fn crop_stamps(rectangles: &mut Vec<Rect>, image_size: Size) {
    for col in 1..=3 {
        for row in 0..3 {
            let mut rect = Rect::new(
                image_size.width / 2 - col * STAMP_WIDTH - LEFT_OFFSET,
                STAMP_HEIGHT * row,
                STAMP_WIDTH,
                STAMP_HEIGHT + STAMP_HEIGHT_LEFT_CORRECTION,
            );

            rect.x = max(0, rect.x); // Ensure x is not less than 0
            rect.y = max(0, rect.y); // Ensure y is not less than 0

            // Check if the rectangle exceeds the image's width
            if rect.x + rect.width > 4096 {
                rect.width = 4096 - rect.x; // Adjust width to fit within the image
            }

            // Check if the rectangle exceeds the image's height
            if rect.y + rect.height > 500 {
                rect.height = 500 - rect.y; // Adjust height to fit within the image
            }


            // Check if the rectangle is within bounds
            if rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= image_size.width && rect.y + rect.height <= image_size.height {
                rectangles.push(rect);
            } else {
                error!("Invalid rectangle: {:?}", rect);
            }
        }
    }

    for col in 0..3 {
        for row in 0..3 {
            let mut rect = Rect::new(
                image_size.width / 2 + RIGHT_OFFSET + col * STAMP_WIDTH,
                (STAMP_HEIGHT + STAMP_HEIGHT_RIGHT_CORRECTION) * (2 - row),
                STAMP_WIDTH,
                STAMP_HEIGHT + STAMP_HEIGHT_RIGHT_CORRECTION,
            );

            rect.x = max(0, rect.x); // Ensure x is not less than 0
            rect.y = max(0, rect.y); // Ensure y is not less than 0

            // Check if the rectangle exceeds the image's width
            if rect.x + rect.width > 4096 {
                rect.width = 4096 - rect.x; // Adjust width to fit within the image
            }

            // Check if the rectangle exceeds the image's height
            if rect.y + rect.height > 500 {
                rect.height = 500 - rect.y; // Adjust height to fit within the image
            }

            // Check if the rectangle is within bounds
            if rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= image_size.width && rect.y + rect.height <= image_size.height {
                rectangles.push(rect);
            } else {
                error!("Invalid rectangle: {:?}", rect);
            }
        }
    }
}


fn draw_filtered_bounding_rectangle(
    contours: &VectorOfVectorOfPoint,
    image: &mut Mat,
) -> Result<Option<Rect>> {
    let mut largest_area = 0.0;
    let mut largest_contour_index = None;

    // Find the largest contour by area
    for (i, contour) in contours.iter().enumerate() {
        let area = imgproc::contour_area(&contour, false)?;
        if area > largest_area {
            largest_area = area;
            largest_contour_index = Some(i);
        }
    }

    let mut x_min = i32::MAX;
    let mut y_min = i32::MAX;
    let mut x_max = i32::MIN;
    let mut y_max = i32::MIN;

    // Calculate bounding box excluding the largest contour
    for (i, contour) in contours.iter().enumerate() {
        if Some(i) == largest_contour_index {
            continue;
        }
        let rect = bounding_rect(&contour)?;
        x_min = x_min.min(rect.x);
        y_min = y_min.min(rect.y);
        x_max = x_max.max(rect.x + rect.width);
        y_max = y_max.max(rect.y + rect.height);
    }

    if x_min < x_max && y_min < y_max {
        let rect = Rect::new(x_min, y_min, x_max - x_min, y_max - y_min);
        // Draw the rectangle on the image
        rectangle(
            image,
            rect,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            LINE_8,
            0,
        )?;
        return Ok(Some(rect));
    }

    Ok(None)
}


fn apply_thresholding(gray: &Mat, threshold_type: &str, value: f64) -> Result<Mat> {
    let mut thresh = Mat::default();
    match threshold_type {
        "binary" => {
            threshold(&gray, &mut thresh, value, 255.0, THRESH_BINARY)?;
        }
        "adaptive_mean" => {
            imgproc::adaptive_threshold(
                &gray,
                &mut thresh,
                255.0,
                ADAPTIVE_THRESH_MEAN_C,
                THRESH_BINARY,
                11,
                2.0,
            )?;
        }
        _ => {
            panic!("Unknown threshold type: {}", threshold_type);
        }
    }
    Ok(thresh)
}

fn apply_noise_removal(thresh: &Mat) -> Result<Mat> {
    // Create an intermediate result
    let mut result = thresh.clone();

    // Apply median blur
    let mut temp = Mat::default();
    imgproc::median_blur(&result, &mut temp, 5)?;
    result = temp;

    // Apply Gaussian blur
    let mut temp = Mat::default();
    imgproc::gaussian_blur(
        &result,
        &mut temp,
        Size::new(15, 15),
        0.0,
        0.0,
        BORDER_DEFAULT,
    )?;
    result = temp;

    apply_thresholding(&result, "binary", 90.0).expect("TODO: panic message");

    let mut temp = Mat::default();
    imgproc::gaussian_blur(
        &result,
        &mut temp,
        Size::new(9, 9),
        0.0,
        0.0,
        BORDER_DEFAULT,
    )?;
    result = temp;
    // Threshold again
    let mut temp = Mat::default();
    imgproc::threshold(&result, &mut temp, 80.0, 255.0, THRESH_BINARY)?;
    result = temp;

    Ok(result)
}


pub fn extract_corners_with_bounding_box(images_path: &str, threshold_type: &str) -> Result<()> {
    let image_files = std::fs::read_dir(images_path).unwrap()

        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .map(|entry| entry.path())
        .collect::<Vec<_>>();


    let mut widths = Vec::new();
    let mut heights = Vec::new();

    for (img_id, image_path) in image_files.iter().enumerate() {
        println!("Processing image {} ", img_id);
        let im = imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        if im.empty() {
            println!("Could not read {:?}. Skipping...", image_path);
            continue;
        }

        let image_size = im.size()?;
        let mut rectangles = vec![];


         crop_stamps(&mut rectangles,image_size);

        for (idx,rect) in rectangles.iter().enumerate() {
            let stamp_roi = im.roi(*rect)?.clone_pointee();

            let mut gray = Mat::default();
            cvt_color(&stamp_roi, &mut gray, COLOR_BGR2GRAY, 0)?;

            let thresh = apply_thresholding(&gray, threshold_type, 90.0)?;
            let processed_thresh = apply_noise_removal(&thresh)?;

            let mut contours = VectorOfVectorOfPoint::new();
            find_contours(
                &processed_thresh,
                &mut contours,
                RETR_TREE,
                CHAIN_APPROX_SIMPLE,
                Point::new(0, 0),
            )?;

            let mut image_with_contours = stamp_roi.clone();
            imgproc::draw_contours(
                &mut image_with_contours,
                &contours,
                -1,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                LINE_8,
                &Mat::default(),
                0,
                Point::new(0, 0),
            )?;

            let rect = draw_filtered_bounding_rectangle(&contours, &mut image_with_contours)?;

            if let Some(rect) = rect {
                // Crop the region
                let cropped_region = stamp_roi.roi(rect)?.clone_pointee();
                widths.push(rect.width);
                heights.push(rect.height);

                let mut resized_region = Mat::default();
                imgproc::resize(
                    &cropped_region,
                    &mut resized_region,
                    Size::new(580, 140),
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;

                // Optionally, display or save the cropped region
                // show_image(&format!("Cropped Region {}", idx), &resized_region)?;


            } else {
                println!("No valid bounding box found for this region.");
            }

            show_image(&format!("Processed ROI {}", idx), &image_with_contours)?;
        }
    }

    if !widths.is_empty() && !heights.is_empty() {
        let (avg_width, std_width, min_width, max_width, median_width) = calculate_statistics(&widths);
        let (avg_height, std_height, min_height, max_height, median_height) = calculate_statistics(&heights);

        println!("Width - Average: {:.2}, Std Dev: {:.2}, Min: {:.2}, Max: {:.2}, Median: {:.2} of {} images",
                 avg_width, std_width, min_width, max_width, median_width, widths.len());
        println!("Height - Average: {:.2}, Std Dev: {:.2}, Min: {:.2}, Max: {:.2}, Median: {:.2} of {} images",
                 avg_height, std_height, min_height, max_height, median_height, heights.len());
    } else {
        println!("No valid dimensions to calculate statistics.");
    }


    Ok(())
}
fn calculate_statistics(values: &Vec<i32>) -> (f64, f64, f64, f64, f64) {
    let len = values.len() as f64;
    let sum: f64 = values.iter().map(|&v| v as f64).sum();
    let mean = sum / len;

    let variance = values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / len;
    let std_dev = variance.sqrt();

    let min = values.iter().map(|&v| v as f64).fold(f64::INFINITY, f64::min);
    let max = values
        .iter()
        .map(|&v| v as f64)
        .fold(f64::NEG_INFINITY, f64::max);

    // Calculate median
    let mut sorted_values = values.clone();
    sorted_values.sort();
    let median = if sorted_values.len() % 2 == 0 {
        let mid = sorted_values.len() / 2;
        (sorted_values[mid - 1] as f64 + sorted_values[mid] as f64) / 2.0
    } else {
        sorted_values[sorted_values.len() / 2] as f64
    };

    (mean, std_dev, min, max, median)
}
