#ifndef PROBABILISTIC_R_EMBEDDED_HPP_GUARD
#define PROBABILISTIC_R_EMBEDDED_HPP_GUARD

class R_embedded {
    public:
        void use(void);
        ~R_embedded();

    private:
        bool used = false;
};

extern R_embedded r_embedded;


#endif

