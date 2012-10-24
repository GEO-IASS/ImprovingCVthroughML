function W=robustveil(a,w)

% No input data, return an untrained classifier.
	if (nargin == 0) | (isempty(a))
		W = mapping(mfilename,{w});
		return;
    else
        try
            W=a*w;
        catch err
            display(err.message)
            W=a*mostcommonc;
        end
    end

    