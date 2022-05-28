Sys.setlocale('LC_ALL','C')
# https://stackoverflow.com/questions/41717781/warning-input-string-not-available-in-this-locale

create_and_copy <- function(from, to, overwrite = recursive, recursive = FALSE, copy.mode = TRUE, copy.date = FALSE) {
  if (!dir.exists(to)) {
    dir.create(to)
    file.copy(from, to, overwrite, recursive, copy.mode, copy.date)
  }
}

dir_except <- function(path, x) {
  # A file or directory is returned if it is
  # not an ancestor or descendent of a file
  # or directory in x.
  
  ancestors.and.descendents <- x
  for (p in x) {
    ancestor <- dirname(p)
    while(!(ancestor == path)) {
      ancestors.and.descendents <- append(ancestors.and.descendents, ancestor)
      ancestor <- dirname(ancestor)
    }
    
    descendents <- dir(
      path = p,
      full.names = TRUE,
      recursive = TRUE,
      include.dirs = TRUE,
      no.. = TRUE
    )
    ancestors.and.descendents <- append(ancestors.and.descendents, descendents)
  }
  
  all.files.dirs <- dir(
    path,
    full.names = TRUE,
    recursive = TRUE,
    include.dirs = TRUE,
    no.. = TRUE
  )
  
  return(sort(setdiff(all.files.dirs, ancestors.and.descendents)))
}
