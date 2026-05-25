"""
This code is about to calculate the scale factor and chi-square value
between experimental data and fitting data.
The scale factor is calculated by the least square method.

The chi-square value is calculated by the following equation.
* χ² = Σ(S₁y_fit + S₂ - y_exp)²
where y_exp is experimental data, y_fit is fitting data.

We can uniquely determine the scale factor by the following equation.
∂χ²/∂S₁ = 0, ∂χ²/∂S₂ = 0

| a b | | S₁ | = | x₁ |  where, a = Σy_fit², b = Σy_fit, x₁ = Σy_fit * y_exp
| c d | | S₂ |   | x₂ |         c = Σy_fit , d = Σ1    , x₂ = Σy_exp

Solution:
  S₁ = (  d * x₁ - b * x₂ )/(a * d - b * c)
  S₂ = ( -c * x₁ + a * x₂ )/(a * d - b * c)
===============================================================================

Of course, we can assume no constant background,
* χ² = Σ(S₁y_fit - y_exp)²

We can uniquely determine the scale factor by the following equation.
∂χ²/∂S₁ = 0

| a | | S₁ | = | x₁ |   where, a  = Σy_fit²,   
                               x₁ = Σy_fit * y_exp

Solution:
  S₁ = x₁/a
===============================================================================
Input:
  ExpData: experimental data
  FitData: fitting data
  TurnOnConst: if true, we will turn on the constant background
"""

function fitSpectrum(ExpData,FitData,TurnOnConst=false)
  println("fitSpectrum in funcFitSpectrum.jl function called")
  if size(ExpData) != size(FitData)
    error("dimension of input data is not the same");
  end

  if TurnOnConst == true
    return fitSpectrum_2Param(ExpData,FitData);
  else
    return fitSpectrum_1Param(ExpData,FitData);
  end

end

function fitSpectrum_1Param(ExpData,FitData)

  sz = size(ExpData);

  a = x₁ = 0;

  for (elem1,elem2) in zip(FitData,ExpData)
    if isnan(elem1) == false && isnan(elem2) == false
      a += elem1 * elem1;  x₁ += elem1 * elem2;
    end
  end

  S₁ = x₁/a;
  S₂ = 0;

  diff_map = zeros(size(ExpData));

  for (it,val) in enumerate(ExpData)
    if isnan(val) == false
      diff_map[it] = abs((S₁ * FitData[it] - ExpData[it]));
    end
  end

  χ² = sum(diff_map .^ 2);

  return S₁, S₂, χ², diff_map;
end

function fitSpectrum_2Param(ExpData,FitData)

  sz = size(ExpData);

  a = 0;  b = 0;  x₁ = 0;
  c = 0;  d = 0;  x₂ = 0;

  for (elem1,elem2) in zip(FitData,ExpData)
    if isnan(elem1) == false && isnan(elem2) == false
      a += elem1 * elem1;  b += elem1;  x₁ += elem1 * elem2;
      c += elem1;          d += 1;      x₂ += elem2;
    end
  end

  S₁ = (  d * x₁ - b * x₂ )/(a * d - b * c)
  S₂ = ( -c * x₁ + a * x₂ )/(a * d - b * c)

  diff_map = zeros(size(ExpData));

  for (it,val) in enumerate(ExpData)
    if isnan(val) == false
      diff_map[it] = abs((S₁ * FitData[it] + S₂ - ExpData[it]));
    end
  end

  χ² = sum(diff_map .^ 2);

  return S₁, S₂, χ², diff_map;
  
end
