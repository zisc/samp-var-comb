libtorch_dict_detail <- function(dict, table, tensor) {
  out <- list(
    dict = dict,
    table = table,
    tensor = tensor
  )
  class(out) <- c("libtorch_dict", class(out))
  return(out)
}

libtorch_dict <- function(object, ...) {
  UseMethod("libtorch_dict")
}

libtorch_dict.default <- function() {
  return(libtorch_dict_detail(
    dict = .Call(C_R_new_libtorch_dict),
    table = list(),
    tensor = list()
  ))
}

libtorch_dict_append <- function(d1, ...) {
  ds <- list(...)
  for (d in ds) {
    if (inherits(d, "libtorch_dict")) {
      .Call(C_R_libtorch_dict_append_dict, d1$dict, d$dict)
      tbll <- length(d1$table)
      d1$tensor <- c(d1$tensor, lapply(d$tensor, function(x){x+tbll}))
      d1$table <- c(d1$table, d$table)
    } else {
      stop("libtorch_dict_append accepts objects of class libtorch_dict only.")
    }
  }
  return(d1)
}

c.libtorch_dict <- function(...) {
  out <- libtorch_dict()
  libtorch_dict_append(out, ...)
  return(out)
}

time_slice <- function(d, begin, end) {
  begin_coerced <- as.integer(begin)
  end_coerced <- as.integer(end)
  if (!inherits(d, "libtorch_dict")) {
    stop("time_slice accepts objects of class libtorch_dict only.")
  }
  return(libtorch_dict_detail(
    dict = .Call(C_R_libtorch_dict_time_slice, d$dict, begin_coerced, end_coerced),
    table = d$table,
    tensor = d$tensor
  ))
}

tibble_with_key <- function(table_in, key_in) {
  if (!inherits(table_in, "tbl_df")) {
    stop("The table must be a tibble.")
  }
  if (!inherits(key_in, "character")) {
    stop("The key must be a string.")
  }
  if (length(key_in) == 0) {
    stop("The key must be non empty.")
  }
  out <- list(
    table = table_in,
    key = key_in
  )
  class(out) <- c("tibble_with_key", class(out))
  return(out)
}

libtorch_dict.tibble_with_key <- function(x, ...) {
  select <- dplyr::select
  arrange <- dplyr::arrange
  all_of <- dplyr::all_of
  arrange <- dplyr::arrange
  across <- dplyr::across
  distinct <- dplyr::distinct
  add_column <- tibble::add_column
  left_join <- dplyr::left_join
  setdiff <- generics::setdiff
  
  dict_x <- libtorch_dict()
  dict_x$table[[1]] <- list(class = "tibble_with_key")
  
  x_tibble <- x$table
  x_key <- x$key
  
  key_out <- x_tibble %>%
    select(all_of(x_key)) %>%
    arrange(across(all_of(x_key))) %>%
    distinct()
  key_out <- key_out %>%
    add_column(libtorch_key = 0:(nrow(key_out)-1))
  dict_x$table[[1]]$key <- key_out
  
  libtorch_tibble <- x_tibble %>%
    left_join(key_out, by = all_of(x_key))
  libtorch_tibble <- libtorch_tibble %>%
    select(setdiff(colnames(libtorch_tibble), x_key))
  
  measured_col_names <- setdiff(colnames(x_tibble), x_key)
  dict_x$tensor <- as.list(rep(1, length(measured_col_names)))
  names(dict_x$tensor) <- measured_col_names
  
  .Call(C_R_libtorch_dict_append_list,
    dict_x$dict,
    as.list(libtorch_tibble),
    "libtorch_key"
  )
  
  return(libtorch_dict_append(
    dict_x,
    libtorch_dict(...)
  ))
}

tibble_with_indices <- function(table_in, ...) {
  if (!inherits(table_in, "tbl_df")) {
    stop("The table must be a tibble.")
  }
  indices <- list(...)
  if (length(indices) == 0) {
    stop("Indices were not provided.")
  }
  for (i in indices) {
    if (!inherits(i, "character")) {
      stop("Indices must be character vectors.")
    }
  }
  return(structure(
    list(
      table = table_in,
      indices = indices
    ),
    class = c("tibble_with_indices", "list")
  ))
}

libtorch_dict.tibble_with_indices <- function(x, ...) {
  setdiff <- generics::setdiff
  select <- dplyr::select
  all_of <- dplyr::all_of
  arrange <- dplyr::arrange
  across <- dplyr::across
  distinct <- dplyr::distinct
  add_column <- tibble::add_column
  left_join <- dplyr::left_join
  
  dict_x <- libtorch_dict()
  dict_x$table[[1]] <- list(class = "tibble_with_indices")
  
  x_tibble <- x$table
  x_indices <- x$indices
  
  all_indices <- unlist(x$indices, use.names = FALSE)
  measured_cols <- setdiff(colnames(x_tibble), all_indices)
  
  libtorch_tibble <- NULL
  dict_x$table[[1]]$index <- vector("list", length(x_indices))
  
  libtorch_index_names <- sapply(1:length(x_indices), function(i){paste0("libtorch_index_",i)})
  
  for (i in 1:length(x_indices)) {
    x_indices_i <- x_indices[[i]]
    libtorch_index_name_i <- libtorch_index_names[[i]]
    
    index_out_i <- x_tibble %>%
      select(all_of(x_indices_i)) %>%
      arrange(across(all_of(x_indices_i))) %>%
      distinct()
    index_out_i <- index_out_i %>%
      add_column("{libtorch_index_name_i}" := 0:(nrow(index_out_i)-1))
    dict_x$table[[1]]$index[[i]] <- index_out_i
    
    libtorch_tibble_i <- x_tibble %>%
      left_join(index_out_i, by = all_of(x_indices_i)) %>%
      select(all_of(libtorch_index_name_i))
    
    if (is.null(libtorch_tibble)) {
      libtorch_tibble <- libtorch_tibble_i
    } else {
      libtorch_tibble <- cbind(libtorch_tibble, libtorch_tibble_i)
    }
  }
  
  names(dict_x$table[[1]]$index) <- libtorch_index_names
  
  libtorch_tibble <- cbind(libtorch_tibble, select(x_tibble, all_of(measured_cols)))
  
  dict_x$tensor <- as.list(rep(1, length(measured_cols)))
  names(dict_x$tensor) <- measured_cols
  
  .Call(C_R_libtorch_dict_append_list,
    dict_x$dict,
    as.list(libtorch_tibble),
    libtorch_index_names
  )
  
  return(libtorch_dict_append(
    dict_x,
    libtorch_dict(...)
  ))
}

libtorch_dict.tbl_ts <- function(x, ...) {
  index_var <- tsibble::index_var
  key_vars <- tsibble::key_vars
  fill_gaps <- tsibble::fill_gaps
  select <- dplyr::select
  all_of <- dplyr::all_of
  distinct <- dplyr::distinct
  arrange <- dplyr::arrange
  across <- dplyr::across
  add_column <- tibble::add_column
  left_join <- dplyr::left_join
  setdiff <- generics::setdiff
  
  if (!is_regular(x)) {
    stop("libtorch_dict is only supported for regularly spaced tsibbles.")
  }
  
  dict_x <- libtorch_dict()
  dict_x$table[[1]] <- list(class = "tbl_ts")
  
  x_tibble <- as_tibble(x)
  x_index <- index_var(x)
  x_key <- key_vars(x)
  
  x_index_values_tibble <- x %>%
    fill_gaps() %>%
    as_tibble() %>%
    select(all_of(x_index)) %>%
    distinct() %>%
    arrange(across(all_of(x_index)))
  x_index_values_tibble <- x_index_values_tibble %>%
    add_column(libtorch_index = 0:(nrow(x_index_values_tibble)-1))
  
  x_index_values_vector <- x_index_values_tibble[[x_index]]
  x_index_min <- min(x_index_values_vector)
  
  if (max(x_index_values_vector) == x_index_min) {
    stop("libtorch_dict is only supported for data with more than one element along the tsibble index.")
  }
  
  dict_x$table[[1]]$index <- list(
    name = x_index,
    first = x_index_min,
    delta = x_index_values_vector[2]-x_index_values_vector[1]
  )
  
  libtorch_tibble <- x_tibble %>%
    left_join(x_index_values_tibble, by = all_of(x_index))
  libtorch_tibble <- libtorch_tibble %>%
    select(setdiff(colnames(libtorch_tibble), x_index))
  libtorch_index_names <- "libtorch_index"
  tsibble_index_names <- x_index
  
  if (length(x_key) > 0) {
    key_out <- x_tibble %>%
      select(all_of(x_key)) %>%
      arrange(across(all_of(x_key))) %>%
      distinct()
    key_out <- key_out %>%
      add_column(libtorch_key = 0:(nrow(key_out)-1))
    dict_x$table[[1]]$key <- key_out
    
    libtorch_tibble <- libtorch_tibble %>%
      left_join(key_out, by = all_of(x_key))
    libtorch_tibble <- libtorch_tibble %>%
      select(setdiff(colnames(libtorch_tibble), x_key))
    libtorch_index_names <- c("libtorch_key", libtorch_index_names)
    tsibble_index_names <- c(x_key, tsibble_index_names)
  }
  
  measured_col_names <- setdiff(colnames(x), tsibble_index_names)
  dict_x$tensor <- as.list(rep(1, length(measured_col_names)))
  names(dict_x$tensor) <- measured_col_names
  
  .Call(C_R_libtorch_dict_append_list,
    dict_x$dict,
    as.list(libtorch_tibble),
    libtorch_index_names
  )
  
  return(libtorch_dict_append(
    dict_x,
    libtorch_dict(...)
  ))
}

as_libtorch_dict <- function(object, ...) {
  UseMethod("as_libtorch_dict")
}

as_libtorch_dict.tbl_ts <- function(...) {
  return(libtorch_dict(...))
}

libtorch_dict_to_tables <- function(object, ...) {
  UseMethod("libtorch_dict_to_tables")
}

libtorch_dict_get_blank_table_tsibble <- function(libtorch_dict_table) {
  setdiff <- generics::setdiff
  select <- dplyr::select
  filter <- dplyr::filter
  all_of <- dplyr::all_of
  mutate <- dplyr::mutate
  
  tsibble_key_col_names <- setdiff(colnames(libtorch_dict_table$key), "libtorch_key")
  return(
    libtorch_dict_table$key %>%
      select(all_of(tsibble_key_col_names)) %>%
      filter(FALSE) %>%
      mutate("{libtorch_dict_table$index$name}" := libtorch_dict_table$index$first) %>%
      as_tsibble(
        key = all_of(tsibble_key_col_names),
        index = all_of(libtorch_dict_table$index$name)
      )
  )
}

libtorch_dict_get_blank_table_tibble <- function(libtorch_dict_table) {
  setdiff <- generics::setdiff
  select <- dplyr::select
  all_of <- dplyr::all_of
  filter <- dplyr::filter
  
  tsibble_key_col_names <- setdiff(colnames(libtorch_dict_table$key), "libtorch_key")
  return(
    libtorch_dict_table$key %>%
      select(all_of(tsibble_key_col_names)) %>%
      filter(FALSE)
  )
}

libtorch_dict_get_blank_table <- function(libtorch_dict_table) {
  if (libtorch_dict_table$class == "tbl_ts") {
    return(libtorch_dict_get_blank_table_tsibble(libtorch_dict_table))
  } else if (libtorch_dict_table$class == "tibble_with_key") {
    return(libtorch_dict_get_blank_table_tibble(libtorch_dict_table))
  }
}

libtorch_list_to_table_tsibble <- function(tensor, tensor_name, libtorch_data) {
  mutate <- dplyr::mutate
  left_join <- dplyr::left_join
  setdiff <- generics::setdiff
  select <- dplyr::select
  all_of <- dplyr::all_of
  
  table_index <- libtorch_data$tensor[[tensor_name]]
  table_meta <- libtorch_data$table[[table_index]]
  
  has_key <- !is.null(table_meta$key)
  has_index <- !is.null(table_meta$index)
  
  i <- 1
  if (has_key) {
    names(tensor)[i] <- "libtorch_key"
    i <- i + 1
  }
  if (has_index) {
    names(tensor)[i] <- "libtorch_index"
    i <- i + 1
  }
  names(tensor)[i] <- tensor_name
  
  table <- as_tibble(tensor)
  table_key_names <- NULL
  if (has_index) {
    table <- table %>%
      mutate("{table_meta$index$name}" := table_meta$index$first + tensor$libtorch_index*table_meta$index$delta)
  }
  if (has_key) {
    table <- table %>%
      left_join(table_meta$key, by = "libtorch_key")
    table_key_names <- setdiff(colnames(table_meta$key), "libtorch_key")
  }
  table <- table %>%
    select(all_of(c(table_key_names, table_meta$index$name, tensor_name))) %>%
    as_tsibble(
      key = all_of(table_key_names),
      index = all_of(table_meta$index$name)
    )
  
  return(table)
}

libtorch_list_to_table_tibble_with_key <- function(tensor, tensor_name, libtorch_data) {
  setdiff <- generics::setdiff
  full_join <- dplyr::full_join
  select <- dplyr::select
  all_of <- dplyr::all_of
  
  table_index <- libtorch_data$tensor[[tensor_name]]
  table_meta <- libtorch_data$table[[table_index]]
  names(tensor) <- c("libtorch_key", tensor_name)
  table_key_names <- setdiff(colnames(table_meta$key), "libtorch_key")
  table <- tensor %>%
    as_tibble() %>%
    full_join(table_meta$key, by = "libtorch_key") %>%
    select(all_of(c(table_key_names, tensor_name)))
  return(table)
}

libtorch_list_to_table_tibble_with_indices <- function(tensor, tensor_name, libtorch_data) {
  mutate <- dplyr::mutate
  left_join <- dplyr::left_join
  select <- dplyr::select
  all_of <- dplyr::all_of
  setdiff <- generics::setdiff
  
  table_index <- libtorch_data$tensor[[tensor_name]]
  table_meta <- libtorch_data$table[[table_index]]
  libtorch_index_names <- names(table_meta$index)
  names(tensor) <- c(libtorch_index_names, tensor_name)
  table <- as_tibble(tensor)
  for (i in 1:length(table_meta$index)) {
    libtorch_index_names_i <- libtorch_index_names[i]
    index_meta_i <- table_meta$index[[i]]
    if (max(table[[libtorch_index_names_i]]) > max(index_meta_i[[libtorch_index_names_i]])) {
      R_idx_name <- c(colnames(index_meta_i), libtorch_index_names_i)
      if (R_idx_name > 1) {
        stop("Unfamiliar indices returned by a libtorch_dict.")
      }
      index_meta_i_R_idx <- index_meta_i[[R_idx_name]]
      first <- index_meta_i_R_idx[1]
      delta <- index_meta_i_R_idx[2] - first
      if (!all.equal(rep(delta, nrow(index_meta_i)-1), diff(index_meta_i_R_idx))) {
        stop("Unfamiliar indices returned by a libtorch_dict.")
      }
      table <- table %>%
        mutate("{R_idx_name}" := first + tensor[[libtorch_index_names_i]]*delta)
    } else {
      table <- table %>%
        left_join(index_meta_i, by = libtorch_index_names_i)
    }
  }
  table <- table %>%
    select(all_of(setdiff(colnames(table), libtorch_index_names)))
  return(table)
}

libtorch_list_to_tables <- function(libtorch_list, libtorch_data) {
  full_join <- dplyr::full_join
  setdiff <- generics::setdiff
  
  raw <- libtorch_list
  out <- vector("list", length(libtorch_data$table))
  for (i in 1:length(raw)) {
    tensor_name <- names(raw[i])
    tensor <- raw[[i]]
    table_index <- libtorch_data$tensor[[tensor_name]]
    table_meta <- libtorch_data$table[[table_index]]
    table <- NULL
    if (table_meta$class == "tbl_ts") {
      table <- libtorch_list_to_table_tsibble(tensor, tensor_name, libtorch_data)
    } else if (table_meta$class == "tibble_with_key") {
      table <- libtorch_list_to_table_tibble_with_key(tensor, tensor_name, libtorch_data)
    } else if (table_meta$class == "tibble_with_indices") {
      table <- libtorch_list_to_table_tibble_with_indices(tensor, tensor_name, libtorch_data)
    } else {
      stop("Class unsupported.")
    }
    if (is.null(out[[table_index]])) {
      out[[table_index]] <- table
    } else {
      out [[table_index]] <- out[[table_index]] %>%
        full_join(table, by = setdiff(colnames(table), tensor_name))
    }
  }
  return(out[!sapply(out, is.null)])
}

libtorch_dict_to_tables.libtorch_dict <- function(libtorch_data) {
  libtorch_list_to_tables(
    .Call(C_R_libtorch_dict_to_list, libtorch_data$dict),
    libtorch_data
  )
}
