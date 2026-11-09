# install packages
packages <- c("dplyr")
# check which packages are not installed
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
# install missing ones
if (length(missing_packages) > 0) {
    install.packages(missing_packages, repos = "https://cloud.r-project.org")
    }
cat("R packages ready\n")