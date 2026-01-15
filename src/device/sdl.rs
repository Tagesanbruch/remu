// SDL2 Backend for REMU

use crate::generated::config::*;

#[cfg(feature = "device")]
use sdl2::Sdl;
#[cfg(feature = "device")]
use sdl2::video::Window;
#[cfg(feature = "device")]
use sdl2::render::{Canvas, Texture};
#[cfg(feature = "device")]
use sdl2::pixels::PixelFormatEnum;
#[cfg(feature = "device")]
use sdl2::event::Event;
#[cfg(feature = "device")]
use std::sync::Mutex;
#[cfg(feature = "device")]
use lazy_static::lazy_static;
#[cfg(feature = "device")]
use crate::common::RemuState;
#[cfg(feature = "device")]
use crate::utils::set_state;

// Usage of UnsafeSendSync to allow global statics for SDL types
// SDL2 types are !Send / !Sync because they are thread-bound (main thread),
// but we only access them from the main thread in REMU.

#[cfg(feature = "device")]
struct UnsafeSendSync<T>(T);

#[cfg(feature = "device")]
unsafe impl<T> Send for UnsafeSendSync<T> {}
#[cfg(feature = "device")]
unsafe impl<T> Sync for UnsafeSendSync<T> {}

#[cfg(feature = "device")]
struct TextureState {
    texture: Option<Texture<'static>>, 
}
#[cfg(feature = "device")]
unsafe impl Send for TextureState {}
#[cfg(feature = "device")]
unsafe impl Sync for TextureState {}

#[cfg(feature = "device")]
lazy_static! {
    static ref SDL_CONTEXT: Mutex<Option<UnsafeSendSync<Sdl>>> = Mutex::new(None);
    static ref SDL_VIDEO: Mutex<Option<UnsafeSendSync<sdl2::VideoSubsystem>>> = Mutex::new(None);
    static ref SDL_CANVAS: Mutex<Option<UnsafeSendSync<Canvas<Window>>>> = Mutex::new(None);
    static ref SDL_EVENT: Mutex<Option<UnsafeSendSync<sdl2::EventPump>>> = Mutex::new(None);
    static ref SDL_TEXTURE: Mutex<TextureState> = Mutex::new(TextureState { texture: None });
}

#[cfg(feature = "device")]
pub fn init_sdl() {
    if !DEVICE || !HAS_VGA { return; }
    
    // Init SDL
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    
    let width = VGA_WIDTH;
    let height = VGA_HEIGHT;
    
    let window = video_subsystem.window("RISC-V32 REMU", width, height)
        .position_centered()
        .build()
        .unwrap();
        
    let mut canvas = window.into_canvas().build().unwrap();
    canvas.clear();
    canvas.present();
    
    let event_pump = sdl_context.event_pump().unwrap();
    let texture_creator = canvas.texture_creator();
    
    // Hack to store texture with static lifetime
    let static_creator: &'static _ = Box::leak(Box::new(texture_creator));
    let mut texture = static_creator.create_texture_streaming(
        PixelFormatEnum::ARGB8888, width, height).unwrap();
    
    // Disable blending so alpha=0 doesn't make pixels transparent (black)
    texture.set_blend_mode(sdl2::render::BlendMode::None);
    
    // Store in globals
    *SDL_CONTEXT.lock().unwrap() = Some(UnsafeSendSync(sdl_context));
    *SDL_EVENT.lock().unwrap() = Some(UnsafeSendSync(event_pump));
    *SDL_VIDEO.lock().unwrap() = Some(UnsafeSendSync(video_subsystem));
    *SDL_CANVAS.lock().unwrap() = Some(UnsafeSendSync(canvas));
    
    *SDL_TEXTURE.lock().unwrap() = TextureState { 
        texture: Some(texture) 
    };
}

#[cfg(feature = "device")]
pub fn update_screen(vmem: &[u8]) {
    if !HAS_VGA || !VGA_SHOW_SCREEN { return; }
    
    let mut canvas_lock = SDL_CANVAS.lock().unwrap();
    if let Some(wrapper) = canvas_lock.as_mut() {
        let canvas = &mut wrapper.0;
        let mut texture_lock = SDL_TEXTURE.lock().unwrap();
        if let Some(texture) = texture_lock.texture.as_mut() {
            let width = VGA_WIDTH;
            let height = VGA_HEIGHT;
            let size = (width * height * 4) as usize;
            
            if vmem.len() >= size {
                texture.update(None, &vmem[0..size], width as usize * 4).unwrap();
                canvas.clear();
                canvas.copy(texture, None, None).unwrap();
                canvas.present();
            }
        }
    }
}

#[cfg(feature = "device")]
pub fn poll_events() {
    let mut event_lock = SDL_EVENT.lock().unwrap();
    if let Some(wrapper) = event_lock.as_mut() {
        let pump = &mut wrapper.0;
        for event in pump.poll_iter() {
            match event {
                Event::Quit {..} => {
                    set_state(RemuState::Stop);
                    // std::process::exit(0); // Don't exit here, let main loop handle it
                },
                _ => {}
            }
        }
    }
}

// Stubs for non-device builds
#[cfg(not(feature = "device"))]
pub fn init_sdl() {}

#[cfg(not(feature = "device"))]
pub fn update_screen(_vmem: &[u8]) {}

#[cfg(not(feature = "device"))]
pub fn poll_events() {}
