library("tools")

source("file_manipulation.R")

# Populates variables cache.dir, src.dir, test.dir and test.deps.dir,
# and the cache.dependencies function.
source("directories.R")

package.name <- "probabilistic"

src.dir <- file.path(package.name, "src")
deps.dir <- file.path(src.dir, "external")
if (!dir.exists(deps.dir)) {
  dir.create(deps.dir)
}

# Boost R Package Dependency and Testing
boost.cache.path <- cache.dependency(
  url = "https://boostorg.jfrog.io/artifactory/main/release/1.69.0/source/boost_1_69_0.tar.gz"
)
boost.cache.path.boost <- file.path(boost.cache.path, "boost")
boost.cache.path.libs <- file.path(boost.cache.path, "libs")
boost.dep.path <- file.path(src.dir, "boost")  # Keep at /src to prevent paths over 100 chars.
boost.dep.path.boost <- file.path(boost.dep.path, "boost")
boost.dep.path.libs <- file.path(boost.dep.path, "libs")
create_and_copy(  # Copy <deps.dir>/boost/ level files and folders.
  from = paste0(
    boost.cache.path,
    .Platform$file.sep,
    c(
      "LICENSE_1_0.txt"
    )
  ),
  to = boost.dep.path,
  recursive = TRUE
)
if (!dir.exists(boost.dep.path.boost)) {
  dir.create(boost.dep.path.boost)
}
# R package BH is a LinkingTo library,
# but doesn't include all boost libraries. Libraries that are provided by the BH package
# could be removed from the final prepared R package if LinkingTo BH.
file.copy(
  from = boost.cache.path.boost,
  to = boost.dep.path,
  overwrite = FALSE,
  recursive = TRUE
)
create_and_copy(  # Copy only the boost source files that are required.
  from = paste0(
    boost.cache.path.libs,
    .Platform$file.sep,
    c(
      "atomic",
      "chrono",
      "date_time",
      "filesystem",
      "locale",
      "log",
      "random",
      "regex",
      "system",
      "thread"
    )
  ),
  to = boost.dep.path.libs,
  recursive = TRUE
)

# Boost for Testing Only
boost.test.path <- file.path(test.deps.dir, "boost")
if (!dir.exists(boost.test.path)) { dir.create(boost.test.path); }
boost.test.path.boost <- file.path(boost.test.path, "boost")
boost.test.path.libs <- file.path(boost.test.path, "libs")
create_and_copy(
  from = file.path(
    boost.cache.path.boost,
    c(
      "test",
      "timer",
      "timer.hpp"
    )
  ),
  to = boost.test.path.boost,
  recursive = TRUE
)
create_and_copy(
  from = file.path(
    boost.cache.path.libs,
    c(
      "test",
      "timer"
    )
  ),
  to = boost.test.path.libs,
  recursive = TRUE
)

# pytorch
# Install python packages needed for compilation even if source code has
# already been downloaded. This is needed for the docker container.
# pip3 has its own cache, so this doesn't take too long if installation
# of these python dependencies has already occurred.
system2(
  "pip3",
  args = c(
    "install",
    "astunparse",
    "numpy",
    "ninja",
    "pyyaml",
    "mkl",
    "mkl-include",
    "setuptools",
    "cmake",
    "cffi",
    "typing",
    "typing_extensions",
    "future",
    "six",
    "requests",
    "dataclasses"
  )
)
if (!dir.exists(file.path("probabilistic", "src", "pytorch"))) {
  gwd <- getwd()
  setwd(file.path("probabilistic", "src"))
  system2("git", c("clone", "--recursive", "https://github.com/pytorch/pytorch.git"))
  setwd(file.path("pytorch"))
  system2("git", c("checkout", "tags/v1.11.0"))
  system2("git", c("submodule", "sync"))
  system2("git", c("submodule", "update", "--init", "--recursive", "--jobs", "0"))
  cmakelists <- readLines(con = "CMakeLists.txt")
  cmakelists <- append(cmakelists, "include(\"../pytorch_header.cmake\")", after = 0)
  cmakelists <- append(cmakelists, "include(\"../pytorch_footer.cmake\")")
  writeLines(text = cmakelists, con = "CMakeLists.txt")
  setwd(gwd)
}

faddeeva.cache.path <- file.path(cache.dir, "Faddeeva")
if (!dir.exists(faddeeva.cache.path)) {
  faddeeva.src.path <- file.path(faddeeva.cache.path, "src")
  faddeeva.include.path <- file.path(faddeeva.cache.path, "include")
  dir.create(faddeeva.cache.path)
  dir.create(faddeeva.src.path)
  dir.create(faddeeva.include.path)
  download.file(url = "http://ab-initio.mit.edu/Faddeeva.cc", destfile = file.path(faddeeva.src.path, "Faddeeva.cc"))
  download.file(url = "http://ab-initio.mit.edu/Faddeeva.hh", destfile = file.path(faddeeva.include.path, "Faddeeva.hh"))
}
create_and_copy(
  from = dir(faddeeva.cache.path, full.names = TRUE),
  to = file.path(deps.dir, "Faddeeva"),
  recursive = TRUE
)
