include shared_vars.mk

all:
	$(MAKE) -f "$(R_HOME)/etc$(R_ARCH)/Makeconf" -f "$(SRC)/MakefileNested" all
    
configure:
	$(MAKE) -f "$(R_HOME)/etc$(R_ARCH)/Makeconf" -f "$(SRC)/MakefileNested" configure

clean:
	$(MAKE) -f "$(R_HOME)/etc$(R_ARCH)/Makeconf" -f "$(SRC)/MakefileNested" clean
	rm -rf \
		build \
		probabilistic.Rcheck \
		probabilistic*.tar.gz

