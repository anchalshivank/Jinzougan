mod utils;

use crate::utils::extract_corners_with_bounding_box;
use dotenv::dotenv;
use opencv::prelude::*;


fn main() {
    // Initialize the logger
    dotenv().ok();
    env_logger::init();

    extract_corners_with_bounding_box(
        "sample/camera",
        "adaptive_mean", // Use Gaussian Blur
        // Use Median Blur
        // Use Morphological Opening
        // Use closed contours
    )
    .unwrap();
}
