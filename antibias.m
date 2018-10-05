% The MIT License (MIT)
%
% Copyright (c) 2018 Colin J. Stoneking
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.
%
%

% ANTIBIAS antibias subjects in behavioral training

%   this function infers and counters any Markovian strategy of stimulus-independent responding
%   i.e. any case where we can predict a subject's choice better than chance
%   based on the previous choice and whether the choice was rewarded
%   e.g. side-bias, alternation or win-stay lose-shift


%  arguments:
%   choice  : vector of choices on previous trials 
%           left-side needs to be coded as 0, right-side as 1
%   outcome : vector of outcomes on previous trials
%           reward needs to be coded as 2, incorrect as 1
%           all other values mean the trial is ignored
%   window  :  sd of Gaussian window with which we weight trial history
%   alpha   :  how harshly/softly to apply antipattern
%           alpha=1: harshest possible
%           larger alpha -> less harsh, more probabilistic -> 
%             avoid possible weird effects due to the effect of the antipattern itself being perfectly predictable
%			alpha = 1.75 seems to work reasonably well

%   returns:
%   strategy     : estimated strategy on a [-0.5, 0.5] scale
%   PChooseRight : estimated probability of choosing right-side on next trial 
%				(predicted based on last choice and outcome, using strategy inferred from trial history)
%   PForceRight  : probability that we should force the next trial to be right-side, to counter the subject's strategy
%               this is based on PChooseRight and alpha

%   usage example:
%
%   [strategy, PChooseRight, PForceRight] = ...
%   antipattern(ChosenSide(1:currentTrial), Outcome(1:currentTrial), Window(currentTrial), Alpha(currentTrial));

%   %display some stats on the inferred strategy:
%   disp('Estimated Strategy: (-0.5 to 0.5 scale)');
%   disp(['RightBias = ' num2str(strategy(1))]);
%   disp(['Alternate = ' num2str(strategy(2))]);
%   disp(['WinStayLoseShift = ' num2str(strategy(3))]);

%   %then choose next side in task:
%   if(rand < PForceRight)
%       nextSide = right;
%   else
%       nextSide = left;
%   end

function [strategy, PChooseRight, PForceRight] = antibias(choice, outcome, window, alpha)


%============================================================

reward    = 2;
incorrect = 1;
left = 0;
right = 1;

N        = length(choice);

valid    = outcome==reward | outcome==incorrect;

validPair   = valid(1:(N-1)) & valid(2:N);
NValidPairs = sum(validPair);

if(NValidPairs < 1)
    %there are no valid pairs to base the estimate on
    strategy = [0, 0, 0, 0]';
    PForceRight = 0.5;
    PChooseRight = 0.5;
else

    if(iscolumn(choice))
        choice = choice';
    end
    if(iscolumn(outcome))
        outcome = outcome';
    end

    prevRight   =  choice(1:(N-1))==right;
    prevReward  =  outcome(1:(N-1))==reward;

    choseRight  =  choice(2:N)==right;
    
    prevRight   = prevRight(validPair);
    prevReward  = prevReward(validPair);
    choseRight  = choseRight(validPair);

    %Gaussian weights on previous trials
    %always regularize to an extent of at least 0.1% of data
    %for numeric stability
    x = 10000:-1:0;
    weights = exp(-x.^2/(2*window^2));
    weights = 0.999*(weights/sum(weights));
    dataWeights = weights((length(weights) - NValidPairs + 1):length(weights));
    regularization = (1 - sum(dataWeights))/8;
	%regularization: add extra 'dummy observations' to all cases
	%so this behaves stably even with few data points
    
	%count number of times we had a given previous side - previous reward combination
	%weighted with a Gaussian kernel, with a regularizing term added
    NRR = sum(( prevRight &  prevReward).*dataWeights) + 2*regularization;
    NLR = sum((~prevRight &  prevReward).*dataWeights) + 2*regularization;
    NRU = sum(( prevRight & ~prevReward).*dataWeights) + 2*regularization;
    NLU = sum((~prevRight & ~prevReward).*dataWeights) + 2*regularization;

    %fraction of times the guy chose right-side for a given combination
	%from weighted and regularized counts -> choice history
    RR = (sum(choseRight.*( prevRight &  prevReward).*dataWeights) + regularization)/NRR;
    LR = (sum(choseRight.*(~prevRight &  prevReward).*dataWeights) + regularization)/NLR;
    RU = (sum(choseRight.*( prevRight & ~prevReward).*dataWeights) + regularization)/NRU;
    LU = (sum(choseRight.*(~prevRight & ~prevReward).*dataWeights) + regularization)/NLU;

    %convert fractions to log-likelihood ratio 
    RR = log(RR/(1 - RR));
    LR = log(LR/(1 - LR));
    RU = log(RU/(1 - RU));
    LU = log(LU/(1 - LU));
    
	%orthogonal transform that brings the choice history matrix
	%into a space where the coordinates correspond to meaningful strategies
	%(side-biased, alternating etc.)
    D = 0.5*[[1, 1, 1, 1];  [-1, 1, -1, 1]; [1, -1, -1, 1]; [1, 1, -1, -1]];
    
	%weight choice histories by number of observations, and transform
    strategy = D*[RR*NRR, LR*NLR, RU*NRU, LU*NLU]'/(NRR + NLR + NRU + NLU);

    %regularize by removing the two strategies which contribute least
    reg = 2;
    removed = zeros(length(strategy), 1);
    
    for m = 1:reg
        lowest = min(abs(strategy(~removed)));
        for n = length(strategy):-1:1
           %start removal from the bottom
           %because we suppose a priori that these strategies
           %are the least relevant
           if(~removed(n) && abs(abs(strategy(n)) - lowest) <= 10^(-12))
               strategy(n) = 0;
               removed(n)   = 1;
               break;
           end
        end
    end
    
    %transform the regularized matrix back
    prediction = 1./(1 + exp(-D'*strategy));
    
    %RR_pred = 1/(1 + exp(-(w1*rightBias - w2*alternate + w3*winStayLoseShift)));
    %LR_pred = 1/(1 + exp(-(w1*rightBias + w2*alternate - w3*winStayLoseShift)));
    %RU_pred = 1/(1 + exp(-(w1*rightBias - w2*alternate - w3*winStayLoseShift)));
    %LU_pred = 1/(1 + exp(-(w1*rightBias + w2*alternate + w3*winStayLoseShift)));

    %convert back to probabilities
    %and subtract 0.5 so that "no effect" is at zero
    %rightBias:     
    %-0.5: perfectly left-biased
    %+0.5: perfectly right-biased
    %alternate:
    % 0.5: always switches
    %-0.5: always repeats previous choice over blocks of a large size,
    %    while still switching occasionally so he is not biased overall
    %winStayLoseShift = (1/(1 + exp(-winStayLoseShift)) - 0.5);
    %winStayLoseShift  = exp(winStayLoseShift);
    %-0.5: perfect win-shift-lose-stay
    %+0.5: perfect win-stay-lose-shift
    
    strategy = 1./(1 + exp(-strategy)) - 0.5;

    PChooseRight = 0.5;

    if(choice(N)==right && outcome(N)==reward)
        PChooseRight = prediction(1);
    elseif(choice(N)==right && outcome(N)~=reward)
        PChooseRight = prediction(3);
    elseif(choice(N)==left && outcome(N)==reward)
        PChooseRight = prediction(2);
    elseif(choice(N)==left && outcome(N)~=reward)
        PChooseRight = prediction(4);
    end
    
    PForceRight = 0.5*( 1 - sign(PChooseRight - 0.5)*(2*abs(PChooseRight - 0.5))^(alpha-1) );
 
end


end

