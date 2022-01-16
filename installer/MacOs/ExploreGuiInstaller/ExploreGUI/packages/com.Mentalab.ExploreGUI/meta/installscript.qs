
function Component()
{
    // default constructor
}

Component.prototype.createOperations = function()
{
    // call the base create operations function
    component.createOperations();
    
    if (systemInfo.kernelType === "winnt") {
        component.addElevatedOperation("LineReplace", "@TargetDir@/exploregui.desktop", "Exec=", "Exec=@TargetDir@/ExploreGUI/ExploreGUI");
        component.addElevatedOperation("LineReplace", "@TargetDir@/exploregui.desktop", "Icon=", "Icon=@TargetDir@/MentalabLogo.png");
        component.addElevatedOperation("Move", "@TargetDir@/exploregui.desktop", "/usr/share/applications/exploregui.desktop");
    } else if (systemInfo.kernelType === "darwin") {
        // var args = ["sudo cp", "-R", "@TargetDir@/ExploreGUI.app", "/Applications"];
        // component.addElevatedOperation("Execute", args);
        component.addElevatedOperation("Copy", "@TargetDir@/ExploreGUI.app", "/Applications/ExploreGUI.app");
    }
    
}
