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
                },
                Event::KeyDown { scancode: Some(sc), .. } => {
                    let k = keymap(sc);
                    if k != 0 {
                        crate::device::keyboard::send_key(k, true);
                    }
                },
                Event::KeyUp { scancode: Some(sc), .. } => {
                     let k = keymap(sc);
                     if k != 0 {
                         crate::device::keyboard::send_key(k, false);
                     }
                },
                _ => {}
            }
        }
    }
}

// Map SDL Scancode to AM Key Code
// Map SDL Scancode to AM Key Code
// See nemu/src/device/keyboard.c
#[cfg(feature = "device")]
fn keymap(sc: sdl2::keyboard::Scancode) -> u32 {
    use sdl2::keyboard::Scancode::*;
    use crate::device::keyboard::*;
    
    match sc {
        Escape => AM_KEY_ESCAPE,
        F1 => AM_KEY_F1, F2 => AM_KEY_F2, F3 => AM_KEY_F3, F4 => AM_KEY_F4,
        F5 => AM_KEY_F5, F6 => AM_KEY_F6, F7 => AM_KEY_F7, F8 => AM_KEY_F8,
        F9 => AM_KEY_F9, F10 => AM_KEY_F10, F11 => AM_KEY_F11, F12 => AM_KEY_F12,
        Grave => AM_KEY_GRAVE,
        Num1 => AM_KEY_1, Num2 => AM_KEY_2, Num3 => AM_KEY_3, Num4 => AM_KEY_4,
        Num5 => AM_KEY_5, Num6 => AM_KEY_6, Num7 => AM_KEY_7, Num8 => AM_KEY_8,
        Num9 => AM_KEY_9, Num0 => AM_KEY_0,
        Minus => AM_KEY_MINUS, Equals => AM_KEY_EQUALS, Backspace => AM_KEY_BACKSPACE,
        Tab => AM_KEY_TAB,
        Q => AM_KEY_Q, W => AM_KEY_W, E => AM_KEY_E, R => AM_KEY_R, T => AM_KEY_T, Y => AM_KEY_Y, U => AM_KEY_U, I => AM_KEY_I, O => AM_KEY_O, P => AM_KEY_P,
        LeftBracket => AM_KEY_LEFTBRACKET, RightBracket => AM_KEY_RIGHTBRACKET, Backslash => AM_KEY_BACKSLASH,
        CapsLock => AM_KEY_CAPSLOCK,
        A => AM_KEY_A, S => AM_KEY_S, D => AM_KEY_D, F => AM_KEY_F, G => AM_KEY_G, H => AM_KEY_H, J => AM_KEY_J, K => AM_KEY_K, L => AM_KEY_L,
        Semicolon => AM_KEY_SEMICOLON, Apostrophe => AM_KEY_APOSTROPHE, Return => AM_KEY_RETURN,
        LShift => AM_KEY_LSHIFT,
        Z => AM_KEY_Z, X => AM_KEY_X, C => AM_KEY_C, V => AM_KEY_V, B => AM_KEY_B, N => AM_KEY_N, M => AM_KEY_M,
        Comma => AM_KEY_COMMA, Period => AM_KEY_PERIOD, Slash => AM_KEY_SLASH, RShift => AM_KEY_RSHIFT,
        LCtrl => AM_KEY_LCTRL, Application => AM_KEY_APPLICATION, LAlt => AM_KEY_LALT, Space => AM_KEY_SPACE, RAlt => AM_KEY_RALT, RCtrl => AM_KEY_RCTRL,
        Up => AM_KEY_UP, Down => AM_KEY_DOWN, Left => AM_KEY_LEFT, Right => AM_KEY_RIGHT,
        Insert => AM_KEY_INSERT, Delete => AM_KEY_DELETE, Home => AM_KEY_HOME, End => AM_KEY_END, PageUp => AM_KEY_PAGEUP, PageDown => AM_KEY_PAGEDOWN,
        _ => AM_KEY_NONE,
    }
}

// Stubs for non-device builds
#[cfg(not(feature = "device"))]
pub fn init_sdl() {}

#[cfg(not(feature = "device"))]
pub fn update_screen(_vmem: &[u8]) {}

#[cfg(not(feature = "device"))]
pub fn poll_events() {}
