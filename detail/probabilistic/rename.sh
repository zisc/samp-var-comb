#!/bin/bash

# To rename the package, change LOWERCASE_FIND_REPLACE. The format is:
# "<current package name>/<new package name>".
# After renaming, double check DESCRIPTION and make changes there to
# reflect the new name, if necessary.

LOWERCASE_FIND_REPLACE="covidensemble/probabilistic"
UPPERCASE_FIND_REPLACE="${LOWERCASE_FIND_REPLACE^^}"

find . -type f -not -path "./.git/*" -not -path "*rename.sh*" -print0 | xargs -0 sed -i "s/${LOWERCASE_FIND_REPLACE}/g"

find . -type f -not -path "./.git/*" -not -path "*rename.sh*" -print0 | xargs -0 sed -i "s/${UPPERCASE_FIND_REPLACE}/g"

shopt -s globstar
for i in {1..5}
do
rename "s/${LOWERCASE_FIND_REPLACE}/g" **
done

