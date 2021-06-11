import logo from './logo.svg';
import './App.css';
import React from "react";
import NavigationBar from "./components/nav-bar";
import { BrowserRouter, Switch, Route, Redirect } from "react-router-dom";
import MainMenuBox from "./components/main-menu-box";
import Home from "./pages/home";
import UploadPage from "./pages/upload";
import HistoryPage from "./pages/history";

function App() {
  console.log();
  return (
      <div className="App">
        <BrowserRouter>
          <NavigationBar />
          <div className="pageBody">
            <Switch>
              <Route exact path="/" component={HistoryPage}/>
                {/*<Route exact path="/history" component={HistoryPage}/>*/}
                {/*<Route exact path="/upload" component={UploadPage}/>*/}
            </Switch>
          </div>
        </BrowserRouter>
      </div>
  );
}

export default App;
