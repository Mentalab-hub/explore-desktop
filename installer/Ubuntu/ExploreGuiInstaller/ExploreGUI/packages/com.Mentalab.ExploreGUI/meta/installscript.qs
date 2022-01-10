
function Component()
{
    // default constructor
}

Component.prototype.createOperations = function()
{
    // call the base create operations function
    component.createOperations();
    
    component.addElevatedOperation("LineReplace", "@TargetDir@/exploregui.desktop", "Exec=", "Exec=@TargetDir@/ExploreGUI/ExploreGUI");
    component.addElevatedOperation("LineReplace", "@TargetDir@/exploregui.desktop", "Icon=", "Icon=@TargetDir@/MentalabLogo.png");
    component.addElevatedOperation("Move", "@TargetDir@/exploregui.desktop", "/usr/share/applications/exploregui.desktop");
    
}
