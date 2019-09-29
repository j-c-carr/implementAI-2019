// Write your JavaScript code.
var app1 = angular.module("app1", []);
app1.controller("ctrl1", function($scope, $http) {
  $scope.shownArray = [];
  $scope.test = function() {
    $http({
      method: "GET",
      url: "http://localhost:5000/dummy"
    }).then(function(response) {
      alert(response.data);
    });
  };
  $scope.test();
});
