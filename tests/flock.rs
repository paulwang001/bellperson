#[cfg(test)]
mod tests {
    use fs2::FileExt;
    use std::fs::File;
    use std::path::PathBuf;
    use std::time::Duration;

    fn tmp_path(filename: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(filename);
        p
    }


    #[test]
    fn it_fs_lock(){
        rayon::scope(|s|{

            s.spawn(|ss|{
               let f = File::create(tmp_path("l32")).unwrap();
               f.lock_exclusive().unwrap();
               ss.spawn(|_|{
                   let ff = File::create(tmp_path("l32")).unwrap();
                   while let Err(_e) = ff.try_lock_exclusive(){
                       std::thread::sleep(Duration::from_secs(1));
                   };
               });
               std::thread::sleep(Duration::from_secs(5));
            });
        });
    }
}
