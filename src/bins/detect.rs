use dotenv::dotenv;
use Jinzougan::utils::{extract_corners_with_bounding_box, process_image_without_cropping};

fn main() {
    // Initialize the logger
    dotenv().ok();
    env_logger::init();

    // extract_corners_with_bounding_box(
    //     "sample/camera",
    //     "adaptive_mean", // Use Gaussian Blur
    //     // Use Median Blur
    //     // Use Morphological Opening
    //     // Use closed contours
    // )
    // .unwrap();

    extract_corners_with_bounding_box(
        "/media/anchalshivank/d628748b-bd5d-418e-9d39-e3227a48b39d/rotoffset/Jinzougan/sample/camera",
        "adaptive_mean",
    ).unwrap();
}
