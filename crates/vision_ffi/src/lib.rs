use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn vision_core_version() -> *const c_char {
    b"vision_core_ffi\0".as_ptr() as *const c_char
}
