# The Impact of Sampling Variability on Estimated Combinations of Distributional Forecasts

The purpose of this repository is the reproduction of the paper named "The Impact of Sampling Variability on Estimated Combinations of Distributional Forecasts". All numerical results are reproduced (that is, nothing is "hardcoded"). Windows, MacOS and linux-based operating systems are supported, so long as they are supported by [Docker](https://www.docker.com/). In addition to this repository, a download of roughly 1GB is required to obtain the docker image, which contains all software used to run the analysis. Reproduction takes roughly twenty hours on a machine with an AMD Ryzen 7 2700X Eight-Core cpu. RAM requirements are modest: anything over 4GB more than suffices.

To reproduce the results and the paper, follow these steps.
1. Click on the green "Code" button above, and select "Download ZIP" from the dropdown menu. Save and expand the zip file on your computer into a location of your choice. Navigate to the directory "samp-var-comb-main" produced by expanding the zip. All files that need to be manually run, downloaded or otherwise interacted with will appear in this directory (and not in the further subdirectory "samp-var-comb-main/detail", which contains code and files representing saved progress). This directory contains the subdirectory "detail", and the files "README.md", "reproduce.bat", "reproduce.sh", "SPXTR.csv" and ".gitignore" (".gitignore" may be hidden from view). Sometimes the folder produced by expanding the zip contains this directory inside another of the same name, in which case the inner-most directory named "samp-var-comb-main" is the correct one.
2. Download the S&P500 data with the following steps.
   - Login to Global Financial Data [here](https://globalfinancialdata.com/). If your credentials are rejected and your subscription is managed by an institution, follow their instructions for access.
   - Type "SPXTR" into the primary search bar and press the Enter key. This is the code that Global Financial Data uses to identify the S&P500 index.![SPXTR 1](https://user-images.githubusercontent.com/8504183/171089464-638d8ee2-bb89-4ae0-96b9-83ae881d02f7.png)
   - Click on the orange download icon to the very right of the row that appears in the table under the search bar. ![SPXTR 2](https://user-images.githubusercontent.com/8504183/171089478-5d17cd60-4d96-4d70-bfb9-6bbf2849c80b.png)
   - In the Download Options panel, make the following changes.
     - Download data by year, from year 1988 to year 2021.
     - Select the "Close Only" checkbox in the Data Fields row.
     - The only adjustments to select are "Split Adjusted" and "Inflation Adjusted", with all others off.
     - Under the Currency dropdown select "Source Currency", under the Data Frequency dropdown select "Daily", and under the Data Format dropdown select "European (DD/MM/YYYY)".
     - Select "None" regarding the Data Fill Method.
     - Select a "Stacked" Worksheet Format, and a "CSV" Output Format.
   ![SPXTR Download Options](https://user-images.githubusercontent.com/8504183/171090319-006893ea-f998-41d7-8e6e-2087c5aba714.png)
   - Click the blue "Download" button in the bottom-right of the panel.
   - Save the file in "samp-var-comb-main" as "SPXTR.csv", overwriting the existing file with the same name.
3. Download Docker by following the instructions [here](https://www.docker.com/products/docker-desktop/). If you use a linux-based operating system, you might also have the option of downloading docker via the package manager for your distro.
4. Execute reproduce.bat if on Windows, or reproduce.sh if on linux or MacOS. Depending on how Docker is installed, you might need to run reproduce.bat/reproduce.sh using administrator/sudo permissions. Progress is saved at many checkpoints as the software runs, so in the event of a crash, power outage, or other error, simply repeat this step. Reproduction will resume at the latest check-point, perhaps after half an hour of upfront unsaved computation.
