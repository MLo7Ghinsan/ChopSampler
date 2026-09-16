fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("cupcakke.ico");
        res.compile().unwrap();
    }
}
