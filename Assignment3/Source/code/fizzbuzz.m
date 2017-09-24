
% FizzBuzz example
% print numbers from 1 to 100 each on a newline,
% if the number if divisible by 3 or 5 print "Fizz" or "Buzz" respectively
% instead
% if divisible by 3 and 5 print "FizzBuzz" instead
for int = 1:100
    
    if (mod(int,3) == 0 && mod(int,5) == 0)
        fprintf('FizzBuzz');
    elseif (mod(int,3) == 0) 
        fprintf('Fizz');
    elseif (mod(int,5) == 0)
        fprintf('Buzz');
    else
        fprintf('%d',int);
    end
    
    fprintf('\n');
   
end