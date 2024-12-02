use eframe::{egui, Frame};
use egui::{CentralPanel, Color32, ColorImage, Context};
use opencv::{
    core::Mat,
    imgcodecs,
    imgproc::{cvt_color, COLOR_BGR2RGBA},
    prelude::*,
};
use rfd::FileDialog;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use opencv::core::Vector;
use opencv::imgcodecs::imwrite;
use Jinzougan::utils::crop_stamps;

/// Crop constants
const STAMP_HEIGHT: i32 = 500 / 3;
const STAMP_HEIGHT_LEFT_CORRECTION: i32 = 0;
const STAMP_HEIGHT_RIGHT_CORRECTION: i32 = 5;
const STAMP_WIDTH: i32 = 610;
const LEFT_OFFSET: i32 = 25;
const RIGHT_OFFSET: i32 = 105;

/// Cache for preloading images
pub struct ImageCache {
    cache: HashMap<usize, Mat>,
    eviction_queue: VecDeque<usize>,
    capacity: usize,
}

impl ImageCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            eviction_queue: VecDeque::new(),
            capacity,
        }
    }

    pub fn get_current(&self, current_idx: usize) -> Option<&Mat> {
        self.cache.get(&current_idx)
    }

    pub fn load_image(&mut self, image_idx: usize, image_path: PathBuf) {
        if self.cache.contains_key(&image_idx) {
            println!("Image already in cache: {}", image_idx);
            return;
        }

        match imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR) {
            Ok(image) => {
                println!("Successfully loaded image at index {}", image_idx);
                self.cache.insert(image_idx, image);
                self.eviction_queue.push_back(image_idx);

                if self.eviction_queue.len() > self.capacity {
                    if let Some(evicted_idx) = self.eviction_queue.pop_front() {
                        println!("Evicted image from cache: {}", evicted_idx);
                        self.cache.remove(&evicted_idx);
                    }
                }
            }
            Err(e) => println!("Failed to load image at index {}: {:?}", image_idx, e),
        }
    }
}

struct ClassifierApp {
    source_folder: Option<PathBuf>,
    image_files: Vec<PathBuf>,
    current_image_index: usize,
    current_cropped_images: Vec<(egui::TextureHandle, Mat, bool)>,
    image_cache: ImageCache,
}

impl Default for ClassifierApp {
    fn default() -> Self {
        Self {
            source_folder: None,
            image_files: vec![],
            current_image_index: 0,
            current_cropped_images: vec![],
            image_cache: ImageCache::new(5),
        }
    }
}
impl eframe::App for ClassifierApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        CentralPanel::default().show(ctx, |ui| {
            ui.heading("Synchronous Image Viewer with Cropped Regions");

            if ui.button("Select Source Folder").clicked() {
                if let Some(folder) = FileDialog::new().pick_folder() {
                    self.load_images_from_folder(folder);
                    self.load_current_image(ctx);
                }
            }

            if !self.image_files.is_empty() {
                ui.horizontal(|ui| {
                    if ui.button("Previous").clicked() {
                        if self.current_image_index > 0 {
                            self.save_cropped_images();
                            self.current_image_index -= 1;
                            self.load_current_image(ctx);
                        }
                    }
                    if ui.button("Next").clicked() {
                        if self.current_image_index < self.image_files.len() - 1 {

                            self.save_cropped_images();
                            self.current_image_index += 1;
                            self.load_current_image(ctx);
                        }
                    }
                });

                if !self.current_cropped_images.is_empty() {
                    egui::Grid::new("cropped_image_grid")
                        .spacing([10.0, 10.0]) // Spacing between cells
                        .striped(true)
                        .show(ui, |ui| {
                            for (i, (texture, image, is_correct)) in self.current_cropped_images.iter_mut().enumerate() {
                                ui.vertical(|ui| {
                                    // Display the image
                                    ui.image((texture.id(), texture.size_vec2()));

                                    // Define the button label and color
                                    let label = if *is_correct {
                                        "Mark Incorrect"
                                    } else {
                                        "Mark Correct"
                                    };

                                    let button_color = if *is_correct {
                                        Color32::GREEN // Green for correct
                                    } else {
                                        Color32::RED // Red for incorrect
                                    };

                                    let button = egui::Button::new(label)
                                        .fill(button_color);

                                    // Create the button with the custom style
                                    if ui.add(button).clicked() {
                                        *is_correct = !*is_correct; // Toggle state
                                        println!(
                                            "Image at index {} is now marked as {}",
                                            i,
                                            if *is_correct { "correct" } else { "incorrect" }
                                        );

                                        if let Some(base_path) = &self.source_folder {
                                            // Handle moving files between folders
                                            let old_folder = if *is_correct { "dataset/bad" } else { "dataset/good" };
                                            let new_folder = if *is_correct { "dataset/good" } else { "dataset/bad" };

                                            let file_name = format!("stamp_{}_{}.png", self.current_image_index + 1, i + 1);

                                            let old_path = base_path.join(old_folder).join(&file_name);
                                            let new_path = base_path.join(new_folder).join(&file_name);

                                            // Ensure the target folder exists
                                            if let Err(e) = fs::create_dir_all(base_path.join(new_folder)) {
                                                eprintln!("Failed to create folder {}: {}", new_folder, e);
                                            }

                                            // Move the file if it exists
                                            if old_path.exists() {
                                                if let Err(e) = fs::rename(&old_path, &new_path) {
                                                    eprintln!(
                                                        "Failed to move file from {} to {}: {}",
                                                        old_path.display(),
                                                        new_path.display(),
                                                        e
                                                    );
                                                } else {
                                                    println!(
                                                        "Moved file from {} to {}",
                                                        old_path.display(),
                                                        new_path.display()
                                                    );
                                                }
                                            }
                                        }
                                    }
                                });

                                if (i + 1) % 6 == 0 {
                                    ui.end_row();
                                }
                            }
                        });

                } else {
                    ui.label("Loading cropped regions...");
                }

                if let Some(current_path) = self.current_image_path() {
                    ui.label(format!(
                        "Image {}/{}: {}",
                        self.current_image_index + 1,
                        self.image_files.len(),
                        current_path.display()
                    ));
                }
            } else {
                ui.label("No images loaded. Select a source folder.");
            }
        });
    }
}

impl ClassifierApp {

    fn save_cropped_images(&mut self) {

        let mut base_path = self.source_folder.clone().unwrap();
        for (index, (texture, image, is_correct)) in self.current_cropped_images.iter_mut().enumerate() {

            let folder_name = if *is_correct {"dataset/good"} else {"dataset/bad"};

            let folder_path = base_path.join(folder_name);

            fs::create_dir_all(&folder_path).expect("Failed to create folder");

            let filename = format!("stamp_{}_{}.png", self.current_image_index + 1, index + 1);
            let file_path = folder_path.join(&filename);

            imwrite(&file_path.to_string_lossy(), image, &Vector::default()).unwrap_or_else(|_| panic!("Failed to save image at index {}: {:?}", index, file_path));



        }

    }
    fn update_image_files(&self, index: usize, is_correct: bool) {
        if let Some(source_folder) = &self.source_folder {
            let old_folder = if is_correct { "dataset/bad" } else { "dataset/good" };
            let new_folder = if is_correct { "dataset/good" } else { "dataset/bad" };

            let file_name = format!("stamp_{}_{}.png", self.current_image_index + 1, index + 1);

            let old_path = source_folder.join(old_folder).join(&file_name);
            let new_path = source_folder.join(new_folder).join(&file_name);

            if old_path.exists() {
                fs::rename(&old_path, &new_path).unwrap_or_else(|e| {
                    println!("Failed to move file {}: {}", old_path.display(), e);
                });
            }
        }
    }


    fn load_images_from_folder(&mut self, folder: PathBuf) {
        self.source_folder = Some(folder.clone());
        self.image_files = fs::read_dir(folder)
            .unwrap()
            .filter_map(|entry| {
                let path = entry.unwrap().path();
                if path.is_file()
                    && path
                    .extension()
                    .map_or(false, |ext| ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("png"))
                {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        self.current_image_index = 0;
    }

    fn load_current_image(&mut self, ctx: &Context) {
        self.current_cropped_images.clear(); // Clear previous textures
        if let Some(current_path) = self.current_image_path().cloned() {
            self.image_cache.load_image(self.current_image_index, current_path.clone());

            if let Some(image) = self.image_cache.get_current(self.current_image_index) {

                let stamps = crop_stamps(image.clone());


                for stamp in stamps {

                    let mut rgba = Mat::default();
                    cvt_color(&stamp, &mut rgba, COLOR_BGR2RGBA, 0).unwrap();
                    let rgba_data = rgba.data_bytes().unwrap();


                    let im = ColorImage::from_rgba_unmultiplied(
                        [rgba.cols() as usize, rgba.rows() as usize],
                        rgba_data,
                    );

                    println!("size {:?}", im.size);

                    let texture = ctx.load_texture(format!("cropped_{}", self.current_cropped_images.len()), im, Default::default());
                    self.current_cropped_images.push((texture, stamp.clone(), true));
                }

            }
        }
    }

    fn current_image_path(&self) -> Option<&PathBuf> {
        self.image_files.get(self.current_image_index)
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Synchronous Image Viewer with Cropped Regions",
        options,
        Box::new(|_cc| Ok(Box::new(ClassifierApp::default()))),
    )
}
