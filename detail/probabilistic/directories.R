library(tools)

# Increase the timeout after which a download is abandoned.
options(timeout = max(300, getOption("timeout")))

cache.dir <- "cache"
if (!dir.exists(cache.dir)) {
  dir.create(cache.dir)
}

test.dir <- "test"
test.deps.dir <- file.path(test.dir, "external")
if (!dir.exists(test.deps.dir)) {
  dir.create(test.deps.dir)
}

cache.dependency <- function(url = NULL, unix.url = url, linux.url = unix.url, mac.url = unix.url, win.url = url, save.as = NULL) {
  archive.url <- ""
  
  sysname <- Sys.info()[['sysname']]
  if (sysname == "Linux") {
    archive.url <- linux.url
  } else if (sysname == "Darwin") {
    archive.url <- mac.url
  } else if (sysname == "Windows") {
    archive.url <- win.url
  } else {
    stop("unrecognised operating system")
  }
  
  archive.file.name <- save.as
  if (is.null(archive.file.name)) {
    archive.file.name <- basename(archive.url)
  }
  
  archive.file.path <- paste0(cache.dir, .Platform$file.sep, archive.file.name)
  archive.file.ext <- file_ext(archive.file.name)
  archive.file.name.noext <- file_path_sans_ext(file_path_sans_ext(archive.file.name))
  archive.dir.path <- NULL
  
  if (!file.exists(archive.file.path)) {
    # Without 'mode = "wb"', the downloaded file is corrupted on windows.
    # The download.file function ought to add mode = "wb" on zip files
    # by default (see ?download.file), perhaps a bug in download.file somwhere?
    download.file(url = archive.url, destfile = archive.file.path, mode = "wb")
  }
  
  if (archive.file.ext %in% c("tar","gz", "tgz","bz2","xz")) {
    tar.list <- untar(tarfile = archive.file.path, exdir = cache.dir, list = TRUE) # Get list only...
    archive.dir.name <- strsplit(tar.list[1], split = .Platform$file.sep)[[1]][1]
    archive.dir.path <- paste0(cache.dir, .Platform$file.sep, archive.dir.name)
    if (!dir.exists(archive.dir.path)) {
      untar(tarfile = archive.file.path, exdir = cache.dir, list = FALSE) # now extract.
    }
  } else if (archive.file.ext == "zip") {
    zip.list <- unzip(zipfile = archive.file.path, exdir = cache.dir, list = TRUE) # Get list only...
    archive.dir.name <- strsplit(zip.list$Name[1], split = .Platform$file.sep)[[1]][1]
    archive.dir.path <- paste0(cache.dir, .Platform$file.sep, archive.dir.name)
    if (!dir.exists(archive.dir.path)) {
      unzip(zipfile = archive.file.path, exdir = cache.dir, list = FALSE) # now extract.
    }
  } else {
    archive.dir.path <- archive.file.path
  }
  
  return(archive.dir.path)
}
