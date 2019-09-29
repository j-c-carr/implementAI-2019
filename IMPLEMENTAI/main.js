// Write your JavaScript code.
var app1 = angular.module("app1", []);
app1.controller("ctrl1", function($scope, $http) {
  $scope.shownArray = [];

  $scope.stock1 = {
    Name: "AQN",
    High: "13.850$",
    Low: "13.569$",
    Open: "13.800$",
    Close: "13.649$"
  };
  $scope.stock2 = {
    Name: "ASTI",
    High: "0.0002$",
    Low: "1.00E-04$",
    Open: "1.00E-04$",
    Close: "1.00E-04$"
  };
  $scope.stock3 = {
    Name: "AZRE",
    High: "10.399$",
    Low: "10.300$",
    Open: "10.390$",
    Close: "10.359$"
  };
  $scope.stock4 = {
    Name: "EVSI",
    High: "5.760$",
    Low: "5.519$",
    Open: "5.664$",
    Close: "5.75$"
  };
  $scope.stock5 = {
    Name: "SEDG",
    High: "86.190$",
    Low: "81.089$",
    Open: "85.510$",
    Close: "82.080$"
  };

  $scope.shownArray.push($scope.stock1);
  $scope.shownArray.push($scope.stock2);
  $scope.shownArray.push($scope.stock3);
  $scope.shownArray.push($scope.stock4);
  $scope.shownArray.push($scope.stock5);
});
