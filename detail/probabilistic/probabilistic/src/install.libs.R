shared.object <- file.path(R_PACKAGE_SOURCE, "src", "build", paste0("lib", R_PACKAGE_NAME, SHLIB_EXT))
libs.dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH))
dir.create(libs.dest, recursive = TRUE, showWarnings = FALSE)
file.copy(shared.object, libs.dest, overwrite = TRUE)

libtorch.shared.objects <- list.files(
  path = file.path(R_PACKAGE_SOURCE, "src", "libtorch", "lib"),
  pattern = "\\.so|\\.dylib",
  full.names = TRUE
)
file.copy(libtorch.shared.objects, libs.dest, overwrite = TRUE)

all.shared.objects.dest <- list.files(
  path = libs.dest,
  pattern = "\\.so|\\.dylib",
  full.names = TRUE
)
Sys.chmod(paths = all.shared.objects.dest, mode = "0775")
