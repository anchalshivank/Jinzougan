use std::cmp::max;
use std::fmt::format;
use std::fs;
use dotenv::dotenv;
use log::{error, info};
use opencv::{
    core::{Point, Rect, Scalar},
    highgui,
    imgcodecs::{imread, IMREAD_COLOR},
    imgproc::{self, rectangle, threshold},
    prelude::*,
    types::VectorOfVectorOfPoint,
    Result,
};
use opencv::core::{no_array, normalize, Size, Vector, NORM_MINMAX};
use opencv::imgproc::{adaptive_threshold, cvt_color, draw_contours, find_contours, ADAPTIVE_THRESH_GAUSSIAN_C, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL, RETR_TREE, THRESH_BINARY, THRESH_OTSU};
use opencv::types::VectorOfMat;

const STAMP_HEIGHT: i32 = 500 / 3;
const STAMP_HEIGHT_LEFT_CORRECTION: i32 = 0;
const STAMP_HEIGHT_RIGHT_CORRECTION: i32 = 5;
const STAMP_WIDTH: i32 = 610;
const LEFT_OFFSET: i32 = 25;
const RIGHT_OFFSET: i32 = 105;

fn extract_stamps_using_contours(image_path: &str) -> Result<()> {
    // Load the image in color
    let mut image = imread(image_path, IMREAD_COLOR)?;
    if image.empty() {
        error!("Error: Could not load image. Please check the file path.");
        return Ok(());
    }

    // Convert to grayscale
    let mut gray = Mat::default();
    imgproc::cvt_color(&image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Image dimensions (width, height)
    let image_size = image.size()?;
    let mut rectangles = vec![];
    crop_stamps(&mut rectangles, image_size);


    // Loop over all rectangles (each stamp)
    for (i, rect) in rectangles.iter().enumerate() {
        // Extract the stamp region from the image
        let stamp_roi = image.roi(*rect)?;

        let mut gray = Mat::default();

        cvt_color(&stamp_roi, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut thresh = Mat::default();
        let threshold_value = 150.0; // Set a fixed threshold value

        threshold(
            &gray,
            &mut  thresh,
            threshold_value,
            255.0,
            THRESH_BINARY
        )?;


        let mut hierarchy = Mat::default();

        let mut contours = VectorOfVectorOfPoint::new();


        highgui::imshow(&format!("Grey Stamp with Threshold {}", i), &thresh)?;

        find_contours(
            &thresh,
            &mut contours,
            RETR_TREE,
            CHAIN_APPROX_SIMPLE,
            Point::new(0,0),
        )?;

        let mut contour_image = stamp_roi.clone_pointee();

        let contour_idx = -1;
        let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let thickness = 2;
        let line_type = imgproc::LINE_8;
        let max_level = 0;
        let offset = Point::new(0, 0);

        draw_contours(
            &mut contour_image,
            &contours,
            contour_idx,
            color,
            thickness,
            line_type,
            &mut hierarchy,
            max_level,
            offset,
        )?;

        highgui::imshow(&format!("Stamp with Timestamps {}", i), &contour_image)?;

        highgui::wait_key(0)?;
    }

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

fn main() {
    // Initialize the logger
    dotenv().ok();
    env_logger::init();

    // Path to the image
    // let image_path = "sample/images/ok/2024-07-08_13-13-53_021.png";

    // Extract stamps and find contours
    // if let Err(e) = extract_stamps_using_contours(image_path) {
    //     eprintln!("An error occurred: {:?}", e);
    // }

    let images_path = "sample/camera";

    extract_corners(images_path);

}
fn extract_corners(images_path: &str) {
    let images = fs::read_dir(images_path).unwrap();

    for (i, image) in images.enumerate() {
        let image_path = image.unwrap().path();

        let im = imread(image_path.to_str().unwrap(), IMREAD_COLOR).unwrap();
        highgui::imshow("Original Image", &im).unwrap();

        let mut gray = Mat::default();
        cvt_color(&im, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();
        highgui::imshow("Grayscale Image", &gray).unwrap();

        let mut thresh = Mat::default();
        threshold(&gray, &mut thresh, 150.0, 255.0, THRESH_BINARY).unwrap();
        highgui::imshow("Binary Image", &thresh).unwrap();
        let mut contours = VectorOfMat::new();
        let mut hierarchy = Mat::default();
        find_contours(
            &thresh,
            &mut contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        ).unwrap();

        println!("Number of contours found: {}", contours.len());

        let mut output_image = im.clone();
        draw_contours(
            &mut output_image,
            &contours,
            -1, // Draw all contours
            Scalar::new(0.0, 255.0, 0.0, 0.0), // Green color
            3,                                 // Thickness
            imgproc::LINE_8,
            &mut hierarchy,
            0,
            Point::new(0, 0),
        ).unwrap();

        highgui::imshow(&format!("Contours Image {}", i), &output_image).unwrap();

        highgui::wait_key(0).unwrap();
    }
}
